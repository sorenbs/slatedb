use crate::manifest::Manifest;
use crate::{
    config::GarbageCollectorDirectoryOptions, db_state::SsTableId, error::SlateDBError,
    manifest::store::ManifestStore, tablestore::TableStore,
};
use chrono::{DateTime, Utc};
use log::error;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::filter::retain_allowed_by_gc_filter;
use super::{CachedDirListing, GcFilter, GcStats, GcTask};
use slatedb_common::object_metadata::IdentifiedObjectMetadata;
use std::collections::HashSet;

/// Selects which class of WAL object a [`WalGcTask`] collects.
///
/// Regular WAL SSTs and zero-byte WAL fence objects share the same WAL
/// directory and `SsTableId::Wal` identifier space, but they have separate
/// retention policies. This mode keeps a single task implementation while
/// allowing regular WAL GC and fence WAL GC to run on independent schedules.
#[derive(Debug, Clone, Copy)]
pub(super) enum WalGcMode {
    /// Collect non-empty WAL SSTs that are older than the compacted WAL
    /// boundary, old enough for retention, and unreferenced by active
    /// checkpoint manifests.
    Regular,

    /// Collect zero-byte WAL fence objects under the same safety checks as
    /// regular WAL GC.
    Fence,
}

#[derive(Clone)]
pub(crate) struct WalGcTask {
    manifest_store: Arc<ManifestStore>,
    table_store: Arc<TableStore>,
    stats: Arc<GcStats>,
    wal_options: GarbageCollectorDirectoryOptions,
    mode: WalGcMode,
    gc_filter: Option<Arc<dyn GcFilter>>,
    dir_listing: CachedDirListing<IdentifiedObjectMetadata<SsTableId>>,
}

impl std::fmt::Debug for WalGcTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WalGcTask")
            .field("wal_options", &self.wal_options)
            .field("mode", &self.mode)
            .finish()
    }
}

impl WalGcTask {
    pub(super) fn new(
        manifest_store: Arc<ManifestStore>,
        table_store: Arc<TableStore>,
        stats: Arc<GcStats>,
        wal_options: GarbageCollectorDirectoryOptions,
        mode: WalGcMode,
        gc_filter: Option<Arc<dyn GcFilter>>,
    ) -> Self {
        WalGcTask {
            manifest_store,
            table_store,
            stats,
            wal_options,
            mode,
            gc_filter,
            dir_listing: CachedDirListing::new(),
        }
    }

    fn is_wal_sst_eligible_for_deletion(
        utc_now: &DateTime<Utc>,
        wal_sst: &IdentifiedObjectMetadata<SsTableId>,
        min_age: &chrono::Duration,
        active_manifests: &BTreeMap<u64, Manifest>,
    ) -> bool {
        if utc_now.signed_duration_since(wal_sst.metadata.last_modified) <= *min_age {
            return false;
        }

        let wal_sst_id = wal_sst.id.unwrap_wal_id();
        !active_manifests
            .values()
            .any(|manifest| manifest.has_wal_sst_reference(wal_sst_id))
    }

    fn wal_sst_min_age(&self) -> chrono::Duration {
        chrono::Duration::from_std(self.wal_options.min_age).expect("invalid duration")
    }
}

impl GcTask for WalGcTask {
    /// Collect garbage from the WAL SSTs. This will delete any WAL SSTs that meet
    /// the following conditions:
    ///  - not referenced by an active checkpoint
    ///  - older than the minimum age specified in the options
    ///  - older than the last compacted WAL SST.
    async fn collect(&self, utc_now: DateTime<Utc>) -> Result<usize, SlateDBError> {
        let latest_manifest = self.manifest_store.read_latest_manifest().await?;
        let active_manifests = self
            .manifest_store
            .read_referenced_manifests(latest_manifest.id, &latest_manifest.manifest)
            .await?;
        let last_compacted_wal_sst_id = latest_manifest.manifest.core.replay_after_wal_id;
        let min_age = self.wal_sst_min_age();
        // The candidate inventory may be a cached listing (list_cache_ttl);
        // the deletion anchors above (latest manifest, checkpoint refs,
        // replay boundary) are re-read fresh on every sweep.
        let ssts_to_delete = self
            .dir_listing
            .entries(
                utc_now,
                self.wal_options.list_cache_ttl,
                self.table_store.list_wal_ssts(..),
            )
            .await?
            .into_iter()
            .filter(|wal_sst| wal_sst.id.unwrap_wal_id() < last_compacted_wal_sst_id)
            .filter(|wal_sst| match self.mode {
                // In regular mode, only consider WAL SSTs with size > 0 for deletion.
                WalGcMode::Regular => wal_sst.metadata.size > 0,
                // In fence mode, only consider zero-byte WAL SSTs for deletion.
                WalGcMode::Fence => wal_sst.metadata.size == 0,
            })
            // Respect min_age and any WAL references held by active checkpoint manifests.
            .filter(|wal_sst| {
                Self::is_wal_sst_eligible_for_deletion(
                    &utc_now,
                    wal_sst,
                    &min_age,
                    &active_manifests,
                )
            })
            .collect::<Vec<_>>();
        let ssts_to_delete = retain_allowed_by_gc_filter(&self.gc_filter, ssts_to_delete).await;
        let sst_ids_to_delete = ssts_to_delete
            .into_iter()
            .map(|wal_sst| wal_sst.id)
            .collect::<Vec<_>>();

        if self.wal_options.dry_run && !sst_ids_to_delete.is_empty() {
            log::info!(
                "dry run: skipping {} deletion [count={}]",
                self.resource(),
                sst_ids_to_delete.len()
            );
            if matches!(self.mode, WalGcMode::Fence) {
                log::info!(
                    "WAL fence GC is dry-run by default. This is a conservative setting. \
                    Set wal_fence_options.dry_run=false and use a conservative min_age to enable. \
                    Silence this log with wal_fence_options=None. See #352 for details."
                );
            }
        }
        let found = sst_ids_to_delete.len();
        let mut deleted = HashSet::new();
        for id in sst_ids_to_delete {
            if self.wal_options.dry_run {
                log::debug!(
                    "dry run: would delete {} but skipped [id={:?}]",
                    self.resource(),
                    id
                );
                continue;
            }
            if let Err(e) = self.table_store.delete_sst(&id).await {
                error!("error deleting WAL SST [id={:?}, error={}]", id, e);
            } else {
                deleted.insert(id);
                match self.mode {
                    WalGcMode::Regular => self.stats.gc_wal_count.increment(1),
                    WalGcMode::Fence => self.stats.gc_wal_fence_count.increment(1),
                }
            }
        }
        self.dir_listing.forget(|sst| deleted.contains(&sst.id));

        Ok(found)
    }

    fn resource(&self) -> &str {
        match self.mode {
            WalGcMode::Regular => "WAL",
            WalGcMode::Fence => "WAL fence",
        }
    }
}
