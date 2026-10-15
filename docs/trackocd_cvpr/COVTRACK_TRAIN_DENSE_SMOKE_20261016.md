# COVTrack Train dense-frame smoke registration — 2026-10-16

This stage registers a bounded frontend smoke before execution. It uses the two smallest fixed video IDs (`0` and `1`) from the already approved 407-video structural image plan. The rule is fixed before any model run and is not selected from outcomes.

The input preparer reads only the plan's structural metadata and lists JPEG names under `/data3/liuyeqiang/TAO-Amodal/frames/train`. It does not open `train.json`, read GT/category arrays, decode or hash pixels, access TAO Val/Test, or train a model. Because the approved plan is sparse metadata, the preparer expands each of the two fixed directories to all locally present JPEG frames and emits a self-contained anonymous COVTrack `TaoDataset` JSON with `annotations=[]`.

The only category records in that runtime JSON are the public COVTrack LVIS vocabulary required by the frozen COVTrack parser/prompt checkpoint. They are not TAO labels, Novel vocabulary, or selection labels. Native COVTrack rows remain separate; the TrackOCD physical stream must be category-free and use `source_split=train_predicted`.

The smoke is limited to one currently idle GPU, two Train videos, and no optimizer steps. It is a frontend/input prerequisite, not a formal OCD/MOT evaluation or frontend qualification. No full 407-video inference is registered by this stage.

The execution receipt and normalization audit will be added after the bounded run; raw native output and predicted cache remain local and are not committed.
