export type CameraKind = "webcam" | "rtsp" | "onvif";

export interface CameraConfig {
  id: string;
  name: string;
  kind: CameraKind;
  source: string;
  enabled: boolean;
  detection_enabled: boolean;
  recording_mode: "event_only" | "full" | "live";
  labels: string[];
  min_confidence: Record<string, number>;
  cooldown_seconds: number;
  motion_threshold: number;
  zones: Array<{ id: string; name: string; mode: "include" | "ignore" | "line"; points: Array<[number, number]> }>;
}

export interface EventItem {
  id: string;
  created_at: string;
  local_time: string;
  camera_id: string;
  label: string;
  confidence: number;
  track_id: number | null;
  zone: string | null;
  motion: number | null;
  reviewed: boolean;
  exported: boolean;
  thumbnail_url: string | null;
  clip_url: string | null;
  search_text: string;
}

export interface EventListResponse {
  items: EventItem[];
  total: number;
}

export interface SettingsResponse {
  settings: {
    version: number;
    data_dir: string;
    export_dir: string | null;
    bind: string;
    port: number;
    allow_lan: boolean;
    armed: boolean;
    telemetry_opt_in: boolean;
    onboarding_completed: boolean;
    retention: { days: number; max_gb: number | null };
    label_thresholds: Record<string, number>;
  };
  cameras: CameraConfig[];
  first_run: boolean;
  data_tree: {
    db: string;
    media: string;
    exports: string;
    logs: string;
  };
}

export interface OnvifEndpoint {
  id: string;
  xaddr: string;
  host: string;
  rtsp_candidates: string[];
}
