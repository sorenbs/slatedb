use crate::TransactionalObjectError::CallbackError;
use crate::{
    BoundaryObject, MonotonicId, ObjectCodec, SequencedStorageProtocol, TransactionalObjectError,
};
use async_trait::async_trait;
use futures::StreamExt;
use log::{debug, error, warn};
use object_store::path::Path;
use object_store::Error::AlreadyExists;
use object_store::{
    Error, GetOptions, ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload, UpdateVersion,
};
use parking_lot::Mutex;
use slatedb_common::object_metadata::IdentifiedObjectMetadata;
use std::collections::Bound;
use std::collections::Bound::Unbounded;
use std::ops::RangeBounds;
use std::sync::Arc;

/// Implements `SequencedStorageProtocol<T>` on object storage.
///
/// ## File layout and naming
/// - Objects are stored under a root directory and logical subdirectory provided at
///   construction time (see `ObjectStoreSequencedStorageProtocol::new`).
/// - Each version is a single file whose name is a zero-padded 20-digit decimal id
///   followed by a fixed suffix, e.g. `00000000000000000001.manifest`.
/// - New versions must use the next consecutive id (`current_id + 1`).
/// - We rely on `put_if_not_exists` to enforce CAS at the storage layer. If a file with
///   the same id already exists, the write fails with `ObjectVersionExists`.
///
/// ## Latest-version reads
/// Because ids are dense at creation (every successful write is exactly its
/// predecessor's id + 1), the latest version can be found without listing the
/// directory whenever a previously observed version is known: probe `known + 1`
/// with a GET, walk forward while versions are found, and when the first probe
/// misses, revalidate the anchor with a conditional GET (`If-None-Match`). The
/// instance caches the newest observed version (id, etag, encoded bytes) to
/// serve as that anchor; writes seed it for free. LIST remains the discovery
/// path when the cache is cold, when the anchor has been deleted out from
/// under us (GC), or when the cache is more than [`PROBE_LIMIT`] versions
/// behind (one LIST jumps to the tail; probing would pay one GET per version).
/// Every returned value is freshly fetched or 304-revalidated — the cache is a
/// hint, never trusted for liveness.
pub struct ObjectStoreSequencedStorageProtocol<T> {
    object_store: Arc<dyn ObjectStore>,
    dir_path: Path,
    codec: Box<dyn ObjectCodec<T>>,
    file_suffix: &'static str,
    boundary: Arc<dyn BoundaryObject>,
    latest_cache: Mutex<Option<CachedLatest>>,
}

/// Newest version observed by this protocol instance, kept as encoded bytes so
/// the cache needs no bounds on `T`.
#[derive(Clone)]
struct CachedLatest {
    id: MonotonicId,
    e_tag: Option<String>,
    bytes: bytes::Bytes,
}

/// After walking this many consecutive found-versions past the cached anchor,
/// fall back to one LIST to jump to the tail instead of paying GET-per-version.
const PROBE_LIMIT: usize = 8;

impl<T> ObjectStoreSequencedStorageProtocol<T> {
    pub fn new(
        root_path: &Path,
        object_store: Arc<dyn ObjectStore>,
        subdir: &str,
        file_suffix: &'static str,
        codec: Box<dyn ObjectCodec<T>>,
    ) -> Self {
        let boundary = Arc::new(ObjectStoreBoundaryObject::new(
            root_path,
            object_store.clone(),
            subdir,
        ));
        Self::new_with_boundary(
            root_path,
            object_store,
            subdir,
            file_suffix,
            codec,
            boundary,
        )
    }

    pub fn new_with_boundary(
        root_path: &Path,
        object_store: Arc<dyn ObjectStore>,
        subdir: &str,
        file_suffix: &'static str,
        codec: Box<dyn ObjectCodec<T>>,
        boundary: Arc<dyn BoundaryObject>,
    ) -> Self {
        Self {
            object_store,
            dir_path: root_path.clone().join(subdir),
            codec,
            file_suffix,
            boundary,
            latest_cache: Mutex::new(None),
        }
    }

    fn path_for(&self, id: MonotonicId) -> Path {
        self.dir_path
            .clone()
            .join(format!("{:020}.{}", id.id(), self.file_suffix))
    }

    /// Fetch one version's raw bytes plus etag; `None` if it does not exist.
    async fn fetch_version(
        &self,
        id: MonotonicId,
    ) -> Result<Option<(Option<String>, bytes::Bytes)>, TransactionalObjectError> {
        match self.object_store.get(&self.path_for(id)).await {
            Ok(obj) => {
                let e_tag = obj.meta.e_tag.clone();
                let bytes = obj.bytes().await?;
                Ok(Some((e_tag, bytes)))
            }
            Err(Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(TransactionalObjectError::from(e)),
        }
    }

    /// Advance the cached latest, never moving it backward. Versions are
    /// immutable once created, so equal ids carry identical bytes.
    fn cache_latest(&self, candidate: CachedLatest) {
        let mut cache = self.latest_cache.lock();
        match cache.as_ref() {
            Some(cached) if cached.id >= candidate.id => {}
            _ => *cache = Some(candidate),
        }
    }

    /// Drop the cache iff it still holds `id` (it was deleted out from under us).
    fn invalidate_cached(&self, id: MonotonicId) {
        let mut cache = self.latest_cache.lock();
        if cache.as_ref().is_some_and(|c| c.id == id) {
            *cache = None;
        }
    }

    /// Confirm the anchor version still exists, reusing cached bytes via a
    /// conditional GET when we hold an etag. Returns `None` if it is gone.
    async fn revalidate_anchor(
        &self,
        cur: &CachedLatest,
    ) -> Result<Option<CachedLatest>, TransactionalObjectError> {
        let Some(e_tag) = cur.e_tag.clone() else {
            return Ok(self.fetch_version(cur.id).await?.map(|(e_tag, bytes)| {
                CachedLatest {
                    id: cur.id,
                    e_tag,
                    bytes,
                }
            }));
        };
        let opts = GetOptions {
            if_none_match: Some(e_tag),
            ..GetOptions::default()
        };
        match self.object_store.get_opts(&self.path_for(cur.id), opts).await {
            Err(Error::NotModified { .. }) => Ok(Some(cur.clone())),
            // Created versions are immutable, so a changed etag is unexpected —
            // but trust the store and refresh our copy.
            Ok(obj) => {
                let e_tag = obj.meta.e_tag.clone();
                let bytes = obj.bytes().await?;
                Ok(Some(CachedLatest {
                    id: cur.id,
                    e_tag,
                    bytes,
                }))
            }
            Err(Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(TransactionalObjectError::from(e)),
        }
    }

    /// Try to resolve the latest version from the cached anchor without a LIST.
    /// `Ok(None)` means "no verdict — discover via LIST" (cold cache, anchor
    /// deleted, or tail more than [`PROBE_LIMIT`] versions ahead).
    async fn try_probe_latest(&self) -> Result<Option<CachedLatest>, TransactionalObjectError> {
        let Some(mut cur) = self.latest_cache.lock().clone() else {
            return Ok(None);
        };
        let mut walked = false;
        for _ in 0..PROBE_LIMIT {
            let next = cur.id.next();
            match self.fetch_version(next).await? {
                Some((e_tag, bytes)) => {
                    cur = CachedLatest {
                        id: next,
                        e_tag,
                        bytes,
                    };
                    walked = true;
                }
                None if walked => {
                    // Freshly fetched on this walk — no revalidation needed.
                    self.cache_latest(cur.clone());
                    return Ok(Some(cur));
                }
                None => {
                    return match self.revalidate_anchor(&cur).await? {
                        Some(fresh) => {
                            self.cache_latest(fresh.clone());
                            Ok(Some(fresh))
                        }
                        None => {
                            self.invalidate_cached(cur.id);
                            Ok(None)
                        }
                    };
                }
            }
        }
        // Still finding versions after PROBE_LIMIT steps: remember how far we
        // got, then let LIST jump the rest of the way.
        self.cache_latest(cur);
        Ok(None)
    }

    fn parse_id(&self, path: &Path) -> Result<MonotonicId, TransactionalObjectError> {
        match path.extension() {
            Some(ext) if ext == self.file_suffix => path
                .filename()
                .expect("invalid filename")
                .split('.')
                .next()
                .ok_or_else(|| TransactionalObjectError::InvalidObjectState)?
                .parse()
                .map(MonotonicId::new)
                .map_err(|_| TransactionalObjectError::InvalidObjectState),
            _ => Err(TransactionalObjectError::InvalidObjectState),
        }
    }
}

/// Implements [`BoundaryObject`] on object storage.
///
/// The boundary is stored as a single ASCII-encoded `u64` at
/// `<root_path>/gc/<name>.boundary`. A missing boundary file is treated as `0`
/// even if this process already has a cached boundary observation.
///
/// Successful reads cache the boundary value along with object-store version
/// metadata. Later reads pass the cached ETag as `if-none-match`, allowing stores
/// that support conditional GETs to return `NotModified`; in that case the cached
/// boundary is reused without fetching the object body again.
pub struct ObjectStoreBoundaryObject {
    object_store: Arc<dyn ObjectStore>,
    filepath: Path,
    /// Caches the last observed boundary and object-store version metadata for
    /// conditional reads.
    cache: Mutex<Option<(MonotonicId, UpdateVersion)>>,
}

impl ObjectStoreBoundaryObject {
    pub fn new(root_path: &Path, object_store: Arc<dyn ObjectStore>, name: &str) -> Self {
        Self {
            object_store,
            filepath: root_path
                .clone()
                .join("gc")
                .join(format!("{name}.boundary")),
            cache: Mutex::new(None),
        }
    }

    /// Updates the cached boundary and version metadata without moving the cached
    /// boundary backward.
    ///
    /// ## Arguments
    ///
    /// * `boundary` - The boundary value read from or written to object storage.
    /// * `version` - The object-store version metadata associated with `boundary`.
    ///
    /// ## Returns
    ///
    /// The boundary and optional version metadata that callers should use after
    /// applying the cache's monotonicity rule.
    fn update_cache(
        &self,
        boundary: MonotonicId,
        version: UpdateVersion,
    ) -> (MonotonicId, Option<UpdateVersion>) {
        let mut cache = self.cache.lock();
        if let Some((cached_id, cached_version)) = cache.as_ref() {
            if *cached_id > boundary {
                return (*cached_id, Some(cached_version.clone()));
            }
        }

        *cache = Some((boundary, version.clone()));
        (boundary, Some(version))
    }

    /// Reads the durable boundary, using a conditional GET when cached version
    /// metadata is present.
    async fn read_boundary(
        &self,
    ) -> Result<(MonotonicId, Option<UpdateVersion>), TransactionalObjectError> {
        let cached = self.cache.lock().clone();
        let opts = GetOptions {
            if_none_match: cached
                .as_ref()
                .and_then(|(_, version)| version.e_tag.clone()),
            ..GetOptions::default()
        };

        match self.object_store.get_opts(&self.filepath, opts).await {
            Ok(result) => {
                let version = UpdateVersion {
                    e_tag: result.meta.e_tag.clone(),
                    version: result.meta.version.clone(),
                };
                let bytes = result.bytes().await?;
                let boundary = MonotonicId::new(
                    std::str::from_utf8(&bytes)
                        .map_err(|_| TransactionalObjectError::InvalidObjectState)?
                        .trim()
                        .parse()
                        .map_err(|_| TransactionalObjectError::InvalidObjectState)?,
                );
                Ok(self.update_cache(boundary, version))
            }
            Err(Error::NotModified { .. }) => match self.cache.lock().clone() {
                Some((boundary, version)) => Ok((boundary, Some(version))),
                // NotModified implies we have a cache, since we need the
                // version's ETag for the conditional GET. If cache is missing,
                // treat as invalid state.
                None => {
                    error!(
                        "received NotModified without cache [path={}]",
                        self.filepath
                    );
                    Err(TransactionalObjectError::InvalidObjectState)
                }
            },
            // A missing boundary is treated as zero. Don't use cache because
            // a write that occurred before a boundary file exists is valid even if
            // we later have a cached boundary above it. Assume the object store is
            // infallible here; if the object store loses data, this will return 0
            // when it should panic. But object stores should never lose data.
            Err(Error::NotFound { .. }) => Ok((MonotonicId::new(0), None)),
            Err(e) => Err(TransactionalObjectError::from(e)),
        }
    }

    /// Checks that the given id is above the boundary, returning an error if not.
    ///
    /// ## Arguments
    /// - `id` - The id to check against the boundary.
    /// - `boundary` - The boundary to check against.
    ///
    /// ## Returns
    /// - `Ok(())` if the id is above the boundary.
    /// - `Err(TransactionalObjectError::ObjectVersionExists)` if the id is at or
    ///   below the boundary.
    async fn check_boundary(
        &self,
        id: MonotonicId,
        boundary: MonotonicId,
    ) -> Result<(), TransactionalObjectError> {
        if id <= boundary {
            debug!(
                "object version is behind boundary: id={:?}, boundary={:?}",
                id.id(),
                boundary.id()
            );
            return Err(TransactionalObjectError::ObjectVersionExists);
        }
        Ok(())
    }
}

#[async_trait]
impl BoundaryObject for ObjectStoreBoundaryObject {
    async fn check(&self, id: MonotonicId) -> Result<(), TransactionalObjectError> {
        // Check the cache first to avoid an object store call when the boundary is stable.
        let cached_boundary = self.cache.lock().clone();
        if let Some((boundary, _)) = cached_boundary {
            self.check_boundary(id, boundary).await?;
        }
        // If cache passed, double check against object store using GET If-None-Match.
        let (boundary, _) = self.read_boundary().await?;
        self.check_boundary(id, boundary).await
    }

    async fn advance(&self, boundary: MonotonicId) -> Result<(), TransactionalObjectError> {
        loop {
            // Use the cache if it's available. If it's stale, we'll refresh the cache when
            // we check the `put_result`.
            let cached_boundary = self.cache.lock().clone();
            let (current_boundary, current_version) =
                if let Some((boundary, version)) = cached_boundary {
                    (boundary, Some(version))
                } else {
                    // No cache, so we have to go to the object store.
                    self.read_boundary().await?
                };

            if current_boundary >= boundary {
                return Ok(());
            }

            let put_result = match current_version {
                Some(version) => {
                    self.object_store
                        .put_opts(
                            &self.filepath,
                            PutPayload::from(boundary.id().to_string()),
                            PutOptions::from(PutMode::Update(version)),
                        )
                        .await
                }
                None => {
                    self.object_store
                        .put_opts(
                            &self.filepath,
                            PutPayload::from(boundary.id().to_string()),
                            PutOptions::from(PutMode::Create),
                        )
                        .await
                }
            };

            match put_result {
                Ok(result) => {
                    self.update_cache(boundary, result.into());
                    return Ok(());
                }
                // Try again if the boundary was concurrently updated by another process.
                Err(Error::AlreadyExists { .. } | Error::Precondition { .. }) => {
                    // Refresh the cache so re-attempts always use the fresh boundary.
                    self.read_boundary().await?;
                }
                Err(e) => return Err(TransactionalObjectError::from(e)),
            }
        }
    }
}

#[async_trait]
impl<T: Send + Sync> BoundaryObject for ObjectStoreSequencedStorageProtocol<T> {
    async fn check(&self, id: MonotonicId) -> Result<(), TransactionalObjectError> {
        self.boundary.check(id).await
    }

    async fn advance(&self, boundary: MonotonicId) -> Result<(), TransactionalObjectError> {
        self.boundary.advance(boundary).await
    }
}

#[async_trait]
impl<T: Send + Sync> SequencedStorageProtocol<T> for ObjectStoreSequencedStorageProtocol<T> {
    async fn write_unchecked(
        &self,
        current_id: Option<MonotonicId>,
        new_value: &T,
    ) -> Result<MonotonicId, TransactionalObjectError> {
        let id = current_id
            .map(|id| id.next())
            .unwrap_or(MonotonicId::initial());
        let path = self.path_for(id);
        let bytes = self.codec.encode(new_value);
        let put_result = self
            .object_store
            .put_opts(
                &path,
                PutPayload::from_bytes(bytes.clone()),
                PutOptions::from(PutMode::Create),
            )
            .await
            .map_err(|err| {
                if let AlreadyExists { path: _, source: _ } = err {
                    TransactionalObjectError::ObjectVersionExists
                } else {
                    TransactionalObjectError::from(err)
                }
            })?;
        self.cache_latest(CachedLatest {
            id,
            e_tag: put_result.e_tag,
            bytes,
        });
        Ok(id)
    }

    async fn try_read_latest_unchecked(
        &self,
    ) -> Result<Option<(MonotonicId, T)>, TransactionalObjectError> {
        if let Some(hit) = self.try_probe_latest().await? {
            let value = self.codec.decode(&hit.bytes).map_err(CallbackError)?;
            return Ok(Some((hit.id, value)));
        }
        loop {
            let files = self.list(Unbounded, Unbounded).await?;
            if let Some(file) = files.last() {
                match self.fetch_version(file.id).await? {
                    Some((e_tag, bytes)) => {
                        let value = self.codec.decode(&bytes).map_err(CallbackError)?;
                        self.cache_latest(CachedLatest {
                            id: file.id,
                            e_tag,
                            bytes,
                        });
                        return Ok(Some((file.id, value)));
                    }
                    // File listed but not found. Probably deleted by GC. Retry list/read.
                    // See https://github.com/slatedb/slatedb/issues/1215 for more details.
                    None => {
                        warn!(
                            "listed file missing on read, retrying [location={}]",
                            file.metadata.location,
                        );
                    }
                }
            } else {
                // No files found, so return None
                break;
            }
        }
        Ok(None)
    }

    async fn try_read_unchecked(
        &self,
        id: MonotonicId,
    ) -> Result<Option<T>, TransactionalObjectError> {
        let path = self.path_for(id);
        match self.object_store.get(&path).await {
            Ok(obj) => match obj.bytes().await {
                Ok(bytes) => self.codec.decode(&bytes).map(Some).map_err(CallbackError),
                Err(e) => Err(TransactionalObjectError::from(e)),
            },
            Err(e) => match e {
                Error::NotFound { .. } => Ok(None),
                _ => Err(TransactionalObjectError::from(e)),
            },
        }
    }

    // List files for this object type within an id range
    async fn list(
        &self,
        from: Bound<MonotonicId>,
        to: Bound<MonotonicId>,
    ) -> Result<Vec<IdentifiedObjectMetadata<MonotonicId>>, TransactionalObjectError> {
        let mut files_stream = self.object_store.list(Some(&self.dir_path));
        let mut items = Vec::new();
        let id_range = (from, to);
        while let Some(file) = match files_stream.next().await.transpose() {
            Ok(file) => file,
            Err(e) => return Err(TransactionalObjectError::from(e)),
        } {
            match self.parse_id(&file.location) {
                Ok(id) if id_range.contains(&id) => {
                    items.push(IdentifiedObjectMetadata::from_object_meta(id, file));
                }
                Err(e) => warn!(
                    "unknown file in directory [base={}, location={}, object_store={}, error={:?}]",
                    self.dir_path, file.location, self.object_store, e,
                ),
                _ => {}
            }
        }
        items.sort_by_key(|m| m.id);
        Ok(items)
    }

    // Delete a specific versioned file (no additional validation)
    async fn delete_unchecked(&self, id: MonotonicId) -> Result<(), TransactionalObjectError> {
        let path = self.path_for(id);
        debug!("deleting object [record_path={}]", path);
        self.object_store
            .delete(&path)
            .await
            .map_err(TransactionalObjectError::from)?;
        self.invalidate_cached(id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ObjectStoreBoundaryObject, ObjectStoreSequencedStorageProtocol};
    use crate::tests::{new_store, TestVal, TestValCodec};
    use crate::{
        BoundaryObject, MonotonicId, ObjectCodec, SequencedStorageProtocol,
        SimpleTransactionalObject, TransactionalObject, TransactionalObjectError,
        TransactionalStorageProtocol,
    };
    use chrono::Utc;
    use futures::stream::{self, BoxStream};
    use futures::StreamExt;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::{
        CopyOptions, Error as ObjectStoreError, GetOptions, GetResult, ListResult, MultipartUpload,
        ObjectMeta, ObjectStore, ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload,
        PutResult, Result as ObjectStoreResult, UpdateVersion,
    };
    use std::collections::Bound::{Excluded, Included, Unbounded};
    use std::fmt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex as StdMutex};
    use tokio::sync::Notify;

    /// A flaky object store that simulates a missing file on the first list() call.
    /// On the first call to list(), it returns a file with `missing_id`. On subsequent
    /// calls, it returns a file with `present_id`. This allows testing retry logic in
    /// `try_read_latest` when a listed file is missing on read. This can happen if the
    /// garbage collector deletes a file between the list and get calls.
    #[derive(Debug)]
    struct FlakyListStore {
        inner: InMemory,
        list_calls: AtomicUsize,
        missing_id: u64,
        present_id: u64,
        file_suffix: &'static str,
    }

    impl FlakyListStore {
        fn new(
            inner: InMemory,
            missing_id: u64,
            present_id: u64,
            file_suffix: &'static str,
        ) -> Self {
            Self {
                inner,
                list_calls: AtomicUsize::new(0),
                missing_id,
                present_id,
                file_suffix,
            }
        }

        fn path_for(&self, id: u64) -> Path {
            Path::from(format!("{:020}.{}", id, self.file_suffix))
        }
    }

    impl fmt::Display for FlakyListStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "FlakyListStore")
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for FlakyListStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> ObjectStoreResult<PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> ObjectStoreResult<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> ObjectStoreResult<GetResult> {
            self.inner.get_opts(location, options).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, ObjectStoreResult<Path>>,
        ) -> BoxStream<'static, ObjectStoreResult<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            _prefix: Option<&Path>,
        ) -> BoxStream<'static, ObjectStoreResult<ObjectMeta>> {
            let call = self.list_calls.fetch_add(1, Ordering::SeqCst);
            let id = if call == 0 {
                self.missing_id
            } else {
                self.present_id
            };
            let meta = ObjectMeta {
                location: self.path_for(id),
                last_modified: Utc::now(),
                size: 0,
                e_tag: None,
                version: None,
            };
            stream::iter(vec![Ok(meta)]).boxed()
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> ObjectStoreResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: CopyOptions,
        ) -> ObjectStoreResult<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    #[derive(Debug)]
    struct BlockingNotFoundGet {
        started: Arc<Notify>,
        release: Arc<Notify>,
    }

    #[derive(Debug)]
    struct CountingGetStore {
        inner: InMemory,
        get_opts_calls: AtomicUsize,
        if_none_match_gets: AtomicUsize,
        list_calls: AtomicUsize,
        blocking_not_found: StdMutex<Option<BlockingNotFoundGet>>,
    }

    impl CountingGetStore {
        fn new() -> Self {
            Self {
                inner: InMemory::new(),
                get_opts_calls: AtomicUsize::new(0),
                if_none_match_gets: AtomicUsize::new(0),
                list_calls: AtomicUsize::new(0),
                blocking_not_found: StdMutex::new(None),
            }
        }

        fn block_next_get_opts_with_not_found(&self) -> (Arc<Notify>, Arc<Notify>) {
            let started = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            *self.blocking_not_found.lock().unwrap() = Some(BlockingNotFoundGet {
                started: started.clone(),
                release: release.clone(),
            });
            (started, release)
        }
    }

    impl fmt::Display for CountingGetStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "CountingGetStore")
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for CountingGetStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> ObjectStoreResult<PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> ObjectStoreResult<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> ObjectStoreResult<GetResult> {
            self.get_opts_calls.fetch_add(1, Ordering::SeqCst);
            if options.if_none_match.is_some() {
                self.if_none_match_gets.fetch_add(1, Ordering::SeqCst);
            }
            let blocking = self.blocking_not_found.lock().unwrap().take();
            if let Some(blocking) = blocking {
                blocking.started.notify_one();
                blocking.release.notified().await;
                return Err(ObjectStoreError::NotFound {
                    path: location.to_string(),
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "injected missing boundary",
                    )),
                });
            }
            self.inner.get_opts(location, options).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, ObjectStoreResult<Path>>,
        ) -> BoxStream<'static, ObjectStoreResult<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, ObjectStoreResult<ObjectMeta>> {
            self.list_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> ObjectStoreResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: CopyOptions,
        ) -> ObjectStoreResult<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    #[tokio::test]
    async fn test_boundary_check_allows_missing_boundary() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let boundary =
            ObjectStoreBoundaryObject::new(&Path::from("/root"), object_store, "manifest");

        boundary.check(MonotonicId::new(1)).await.unwrap();
    }

    #[tokio::test]
    async fn test_boundary_advance_creates_boundary_and_rejects_at_or_below_it() {
        let object_store = Arc::new(InMemory::new());
        let boundary =
            ObjectStoreBoundaryObject::new(&Path::from("/root"), object_store.clone(), "manifest");

        boundary.advance(MonotonicId::new(2)).await.unwrap();

        let err = boundary.check(MonotonicId::new(2)).await.unwrap_err();
        assert!(matches!(err, TransactionalObjectError::ObjectVersionExists));
        boundary.check(MonotonicId::new(3)).await.unwrap();

        let raw_boundary = object_store
            .get(&Path::from("/root/gc/manifest.boundary"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!("2", std::str::from_utf8(&raw_boundary).unwrap());
    }

    #[tokio::test]
    async fn test_boundary_advance_is_monotonic() {
        let object_store = Arc::new(InMemory::new());
        let boundary =
            ObjectStoreBoundaryObject::new(&Path::from("/root"), object_store.clone(), "manifest");

        boundary.advance(MonotonicId::new(3)).await.unwrap();
        boundary.advance(MonotonicId::new(2)).await.unwrap();

        let err = boundary.check(MonotonicId::new(3)).await.unwrap_err();
        assert!(matches!(err, TransactionalObjectError::ObjectVersionExists));

        let raw_boundary = object_store
            .get(&Path::from("/root/gc/manifest.boundary"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!("3", std::str::from_utf8(&raw_boundary).unwrap());
    }

    #[tokio::test]
    async fn test_boundary_check_reuses_cache_on_not_modified() {
        let counting_store = Arc::new(CountingGetStore::new());
        let object_store: Arc<dyn ObjectStore> = counting_store.clone();
        let boundary =
            ObjectStoreBoundaryObject::new(&Path::from("/root"), object_store.clone(), "manifest");

        boundary.advance(MonotonicId::new(2)).await.unwrap();

        let if_none_match_gets = counting_store.if_none_match_gets.load(Ordering::SeqCst);
        boundary.check(MonotonicId::new(3)).await.unwrap();

        assert_eq!(
            if_none_match_gets + 1,
            counting_store.if_none_match_gets.load(Ordering::SeqCst)
        );
    }

    #[tokio::test]
    async fn test_boundary_check_rejects_from_cache_without_get() {
        let counting_store = Arc::new(CountingGetStore::new());
        let object_store: Arc<dyn ObjectStore> = counting_store.clone();
        let boundary =
            ObjectStoreBoundaryObject::new(&Path::from("/root"), object_store.clone(), "manifest");

        boundary.advance(MonotonicId::new(2)).await.unwrap();

        let get_opts_calls = counting_store.get_opts_calls.load(Ordering::SeqCst);
        let err = boundary.check(MonotonicId::new(2)).await.unwrap_err();

        assert!(matches!(err, TransactionalObjectError::ObjectVersionExists));
        assert_eq!(
            get_opts_calls,
            counting_store.get_opts_calls.load(Ordering::SeqCst)
        );
    }

    #[tokio::test]
    async fn test_boundary_check_refreshes_cache_when_etag_changes() {
        let counting_store = Arc::new(CountingGetStore::new());
        let object_store: Arc<dyn ObjectStore> = counting_store.clone();
        let first_boundary =
            ObjectStoreBoundaryObject::new(&Path::from("/root"), object_store.clone(), "manifest");
        let second_boundary =
            ObjectStoreBoundaryObject::new(&Path::from("/root"), object_store, "manifest");

        first_boundary.advance(MonotonicId::new(2)).await.unwrap();
        second_boundary.advance(MonotonicId::new(4)).await.unwrap();

        let err = first_boundary.check(MonotonicId::new(4)).await.unwrap_err();

        assert!(matches!(err, TransactionalObjectError::ObjectVersionExists));
    }

    #[tokio::test]
    async fn test_boundary_advance_retries_after_conflicting_boundary_update() {
        let object_store = Arc::new(InMemory::new());
        let first_boundary =
            ObjectStoreBoundaryObject::new(&Path::from("/root"), object_store.clone(), "manifest");
        let second_boundary =
            ObjectStoreBoundaryObject::new(&Path::from("/root"), object_store.clone(), "manifest");

        first_boundary.advance(MonotonicId::new(2)).await.unwrap();
        second_boundary.advance(MonotonicId::new(3)).await.unwrap();

        first_boundary.advance(MonotonicId::new(4)).await.unwrap();

        let raw_boundary = object_store
            .get(&Path::from("/root/gc/manifest.boundary"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!("4", std::str::from_utf8(&raw_boundary).unwrap());
    }

    #[tokio::test]
    async fn test_boundary_read_returns_zero_when_not_found_get_has_cached_boundary() {
        let counting_store = Arc::new(CountingGetStore::new());
        let (started, release) = counting_store.block_next_get_opts_with_not_found();
        let object_store: Arc<dyn ObjectStore> = counting_store;
        let boundary = Arc::new(ObjectStoreBoundaryObject::new(
            &Path::from("/root"),
            object_store,
            "manifest",
        ));

        let read = tokio::spawn({
            let boundary = boundary.clone();
            async move { boundary.read_boundary().await }
        });

        started.notified().await;
        *boundary.cache.lock() = Some((
            MonotonicId::new(2),
            UpdateVersion {
                e_tag: Some("\"etag\"".to_string()),
                version: None,
            },
        ));
        release.notify_one();

        let (observed_boundary, observed_version) = read.await.unwrap().unwrap();
        assert_eq!(MonotonicId::new(0), observed_boundary);
        assert!(observed_version.is_none());
    }

    #[tokio::test]
    async fn test_list_ranges_sorted() {
        let store = new_store();
        let mut sr = SimpleTransactionalObject::<TestVal>::init(
            Arc::clone(&store) as Arc<dyn TransactionalStorageProtocol<TestVal, MonotonicId>>,
            TestVal {
                epoch: 0,
                payload: 1,
            },
        )
        .await
        .unwrap();
        for p in 2..=4u64 {
            let mut dirty = sr.prepare_dirty().unwrap();
            dirty.value = TestVal {
                epoch: 0,
                payload: p,
            };
            sr.update(dirty).await.unwrap();
        }

        let all = store.list(Unbounded, Unbounded).await.unwrap();
        assert_eq!(4, all.len());
        assert!(all.windows(2).all(|w| w[0].id < w[1].id));
        assert_eq!(
            Path::from("/root/test/00000000000000000001.val"),
            all[0].metadata.location
        );
        assert_eq!(
            Path::from("/root/test/00000000000000000004.val"),
            all[3].metadata.location
        );

        let right_bounded = store.list(Unbounded, Excluded(3.into())).await.unwrap();
        assert_eq!(2, right_bounded.len());
        assert_eq!(1, right_bounded[0].id);
        assert_eq!(2, right_bounded[1].id);

        let left_bounded = store.list(Included(3.into()), Unbounded).await.unwrap();
        assert_eq!(2, left_bounded.len());
        assert_eq!(3, left_bounded[0].id);
        assert_eq!(4, left_bounded[1].id);
    }

    #[tokio::test]
    async fn test_try_read_unchecked_missing_returns_none() {
        let store = new_store();

        let missing = store.try_read_unchecked(1.into()).await.unwrap();

        assert!(missing.is_none());
    }

    /// Validate that try_read_latest retries when a listed file is missing on read.
    #[tokio::test]
    async fn test_try_read_latest_retries_missing_listed_file() {
        let expected = TestVal {
            epoch: 7,
            payload: 42,
        };
        let missing_id = 1u64;
        let present_id = 2u64;

        let inner = InMemory::new();
        let codec = TestValCodec;
        let present_path = Path::from(format!("{:020}.val", present_id));
        inner
            .put(
                &present_path,
                PutPayload::from_bytes(codec.encode(&expected)),
            )
            .await
            .unwrap();

        let flaky_store = Arc::new(FlakyListStore::new(inner, missing_id, present_id, "val"));
        let object_store: Arc<dyn ObjectStore> = flaky_store.clone();
        let store = ObjectStoreSequencedStorageProtocol {
            object_store,
            dir_path: Path::default(),
            codec: Box::new(TestValCodec),
            file_suffix: "val",
            boundary: Arc::new(ObjectStoreBoundaryObject::new(
                &Path::from("/root"),
                flaky_store.clone(),
                "test",
            )),
            latest_cache: parking_lot::Mutex::new(None),
        };

        let latest = store.try_read_latest().await.unwrap().unwrap();
        assert_eq!(present_id, latest.0.id());
        assert_eq!(expected, latest.1);
        assert!(
            flaky_store.list_calls.load(Ordering::SeqCst) >= 2,
            "expected try_read_latest to retry after a missing read"
        );
    }

    fn val(payload: u64) -> TestVal {
        TestVal { epoch: 1, payload }
    }

    fn probe_store(
        os: Arc<CountingGetStore>,
    ) -> ObjectStoreSequencedStorageProtocol<TestVal> {
        ObjectStoreSequencedStorageProtocol::new(
            &Path::from("/root"),
            os,
            "test",
            "val",
            Box::new(TestValCodec),
        )
    }

    #[tokio::test]
    async fn test_read_latest_after_own_write_needs_no_list() {
        let os = Arc::new(CountingGetStore::new());
        let store = probe_store(os.clone());

        let id = store.write(None, &val(1)).await.unwrap();
        let baseline_lists = os.list_calls.load(Ordering::SeqCst);
        for _ in 0..3 {
            let (latest_id, latest) = store.try_read_latest().await.unwrap().unwrap();
            assert_eq!(id, latest_id);
            assert_eq!(val(1), latest);
        }
        assert_eq!(
            baseline_lists,
            os.list_calls.load(Ordering::SeqCst),
            "stable polls should probe from the write-seeded cache, not LIST"
        );
    }

    #[tokio::test]
    async fn test_read_latest_probes_forward_over_external_writes() {
        let os = Arc::new(CountingGetStore::new());
        let ours = probe_store(os.clone());
        let theirs = probe_store(os.clone());

        let mut id = ours.write(None, &val(1)).await.unwrap();
        ours.try_read_latest().await.unwrap().unwrap();
        let baseline_lists = os.list_calls.load(Ordering::SeqCst);

        // Another process advances the object a few versions.
        id = theirs.write(Some(id), &val(2)).await.unwrap();
        id = theirs.write(Some(id), &val(3)).await.unwrap();

        let (latest_id, latest) = ours.try_read_latest().await.unwrap().unwrap();
        assert_eq!(id, latest_id);
        assert_eq!(val(3), latest);
        assert_eq!(
            baseline_lists,
            os.list_calls.load(Ordering::SeqCst),
            "a probe walk should absorb external writes without LIST"
        );
    }

    #[tokio::test]
    async fn test_read_latest_far_behind_falls_back_to_one_list() {
        let os = Arc::new(CountingGetStore::new());
        let ours = probe_store(os.clone());
        let theirs = probe_store(os.clone());

        let mut id = ours.write(None, &val(1)).await.unwrap();
        ours.try_read_latest().await.unwrap().unwrap();
        let baseline_lists = os.list_calls.load(Ordering::SeqCst);

        for i in 2..=20 {
            id = theirs.write(Some(id), &val(i)).await.unwrap();
        }

        let (latest_id, latest) = ours.try_read_latest().await.unwrap().unwrap();
        assert_eq!(id, latest_id);
        assert_eq!(val(20), latest);
        assert_eq!(
            baseline_lists + 1,
            os.list_calls.load(Ordering::SeqCst),
            "beyond PROBE_LIMIT the read should jump to the tail with one LIST"
        );

        // The fallback reseeds the cache: the next stable poll probes again.
        ours.try_read_latest().await.unwrap().unwrap();
        assert_eq!(baseline_lists + 1, os.list_calls.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_read_latest_survives_anchor_deletion() {
        let os = Arc::new(CountingGetStore::new());
        let ours = probe_store(os.clone());
        let theirs = probe_store(os.clone());

        let first = ours.write(None, &val(1)).await.unwrap();
        let second = ours.write(Some(first), &val(2)).await.unwrap();
        ours.try_read_latest().await.unwrap().unwrap();

        // GC (another process) removes the version our cache is anchored on.
        theirs.delete_unchecked(second).await.unwrap();

        let (latest_id, latest) = ours.try_read_latest_unchecked().await.unwrap().unwrap();
        assert_eq!(first, latest_id);
        assert_eq!(val(1), latest);
    }
}
