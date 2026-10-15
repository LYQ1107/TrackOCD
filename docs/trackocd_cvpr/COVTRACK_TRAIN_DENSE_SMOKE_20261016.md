# COVTrack Train dense-frame smoke registration — 2026-10-16

This stage registers a bounded frontend smoke before execution. It uses the two smallest fixed video IDs (`0` and `1`) from the already approved 407-video structural image plan. The rule is fixed before any model run and is not selected from outcomes.

The input preparer reads only the plan's structural metadata and lists JPEG names under `/data3/liuyeqiang/TAO-Amodal/frames/train`. It does not open `train.json`, read GT/category arrays, decode or hash pixels, access TAO Val/Test, or train a model. Because the approved plan is sparse metadata, the preparer expands each of the two fixed directories to all locally present JPEG frames and emits a self-contained anonymous COVTrack `TaoDataset` JSON with `annotations=[]`.

The recovered public COVTrack tracker has a hard-coded `filename.index('val')` compatibility assumption. The smoke uses a runtime-only symlink named `val_frame_alias` whose target is exactly the Train frame root, so the model receives Train pixels while the legacy string branch remains executable. No data is copied and no Val files are referenced; the alias target and symlink status are recorded in the input audit.

The only category records in that runtime JSON are the public COVTrack LVIS vocabulary required by the frozen COVTrack parser/prompt checkpoint. They are not TAO labels, Novel vocabulary, or selection labels. Native COVTrack rows remain separate; the TrackOCD physical stream must be category-free and use `source_split=train_predicted`.

The smoke is limited to one currently idle GPU, two Train videos, and no optimizer steps. It is a frontend/input prerequisite, not a formal OCD/MOT evaluation or frontend qualification. No full 407-video inference is registered by this stage.

The bounded run completed in namespace `covtrack_train_dense_smoke_v1_20261016_alias1` on GPU 0. It processed both fixed Train videos (2,342 frames), produced 81,471 native observations and 8,233 normalized physical tracks, and had zero invalid boxes. The physical stream passed the category/text/GT boundary audit with `source_split=train_predicted`; no OCD/MOT metric was run or claimed. The compact receipt is `outputs/trackocd_cvpr/cvpr2/covtrack_train_dense_smoke_v1/run.json`; raw native JSON and predicted cache remain local and are not committed.

The first launch attempt failed closed before child-process creation because the runner looked for the committed input receipt under the runtime output root instead of the registered CVPR receipt root. GPU 0 remained idle and no image/model/GT/Test input was touched. The path binding was corrected in `c4ab254`.

The corrected runner then reached the frozen COVTrack model but stopped at frame 0 because of the public tracker filename assumption. This is an input-path compatibility failure, not a metric or scientific result; no output was accepted. The registered symlink-only Train alias is the minimal compatibility repair.
