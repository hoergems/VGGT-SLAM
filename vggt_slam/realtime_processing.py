"""Shared realtime submap processing, without camera or model imports."""


def retain_fixed_window_overlap(records, overlap_size):
    """Return the records retained after a completed fixed-policy submap."""
    if overlap_size < 0:
        raise ValueError("overlap_size must be non-negative")
    return list(records[-overlap_size:]) if overlap_size > 0 else []


def should_flush_final_fixed_window(num_records, completed_submaps, overlap_size):
    """Whether a fixed-policy buffer contains unprocessed keyframes."""
    if num_records < 0 or completed_submaps < 0 or overlap_size < 0:
        raise ValueError("record count, submap count, and overlap size must be non-negative")
    if completed_submaps == 0:
        return num_records > 0
    return num_records > overlap_size


def process_submap(keyframe_records, solver, model, args, clip_model,
                   clip_preprocess, planning_map_snapshot, data_lock,
                   map_updated_event):
    """Run inference, graph optimization and publication; propagate failures."""
    image_names = [record.image_path for record in keyframe_records]
    print(f"[SLAM] Processing submap ({len(keyframe_records)} frames)...")
    predictions = solver.run_predictions(
        image_names, model, args.max_loops, clip_model, clip_preprocess,
        keyframe_records=keyframe_records,
    )
    with data_lock:
        solver.add_points(predictions)
        solver.graph.optimize()
        if planning_map_snapshot is not None:
            planning_map_snapshot.update(solver)
        if args.vis_map:
            if len(predictions.get("detected_loops", [])) > 0:
                solver.update_all_submap_vis()
            else:
                solver.update_latest_submap_vis()
    if args.vis_map_open3d:
        map_updated_event.set()
    print("[SLAM] Submap done.")
