# BackendGateway: WebSocket Communication Layer

## 1. Responsibilities
This file contains two classes that handle **two-way** real-time communication with the Django backend:
- **ResultPublisher**: Sends pipeline results (tracks, counts, map positions) **TO** the server.
- **SettingsSubscriber**: Listens for config changes (mapping, counting) **FROM** the server.

## 2. ResultPublisher

### Input: `send_frame(frame_data)`
| Parameter | Type | Description |
|---|---|---|
| `frame_data` | `FrameData` (Pydantic) | Contains all per-frame results: bboxes, track_ids, counts, map_positions |

### Output
- JSON payload sent over WebSocket to `RESULTS_WS_URL`.

### Features
- **Auto-connect**: Connects on first `send_frame()` call.
- **Auto-reconnect**: If send fails and `reconnect=True`, automatically reconnects and retries.
- **Bearer Token Auth**: Reads from `API_BEARER_TOKEN` env var or constructor parameter.

### Usage
```python
publisher = ResultPublisher(
    endpoint="ws://server:8000/ws/results/",
    bearer_token="my-token",
)
publisher.send_frame(frame_data)
publisher.close()
```

## 3. SettingsSubscriber

### Responsibilities
- Listens on **2 separate WebSocket channels** (mapping + counting) in background threads.
- When the backend sends a new config, it deserializes the JSON and calls the registered callback.
- The PipelineManager uses these callbacks to **hot-reload** MappingService and CountingService without restarting.

### Features
- **Daemon Threads**: Background listeners that die with the main process.
- **Auto-reconnect**: Reconnects every 3 seconds if the connection drops.
- **Ping/Pong**: `ping_interval=30, ping_timeout=10` to keep the connection alive.

### Usage
```python
subscriber = SettingsSubscriber(
    mapping_ws_url="ws://server:8000/ws/mapping-settings/",
    counting_ws_url="ws://server:8000/ws/counting-settings/",
    on_mapping_update=lambda data: mapping_service.update_mapping(**data),
    on_counting_update=lambda data: counting_service.update_config(**data),
)
subscriber.start()   # Starts 2 background threads
subscriber.stop()    # Stops all threads
```

## 4. Environment Variables
| Variable | Description | Example |
|---|---|---|
| `RESULTS_WS_URL` | WebSocket URL for publishing results | `ws://192.168.1.100:8000/ws/results/` |
| `MAPPING_SETTINGS_WS_URL` | WebSocket URL for mapping config updates | `ws://192.168.1.100:8000/ws/mapping-settings/` |
| `COUNTING_SETTINGS_WS_URL` | WebSocket URL for counting config updates | `ws://192.168.1.100:8000/ws/counting-settings/` |
| `API_BEARER_TOKEN` | Authentication token | `abc123` |

## 5. Dependencies
- `websocket-client` (Python package)
- `loguru` (for structured logging)
