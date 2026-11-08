CREATE TABLE IF NOT EXISTS detections (
    id          bigserial PRIMARY KEY,
    frame_id    text        NOT NULL,
    camera_id   text        NOT NULL,
    object_key  text        NOT NULL,
    detections  jsonb       NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS detections_camera_created_at_idx
    ON detections (camera_id, created_at DESC);
