# Streaming Research Report: LLM Response Streaming and Continuous Inference Output

**Researcher**: Streaming Researcher
**Date**: 2026-01-12
**Status**: Complete

---

## Executive Summary

This report provides a comprehensive analysis of streaming protocols and implementation patterns for delivering LLM responses in real-time applications. Based on extensive research of current best practices, performance benchmarks, and production implementations, the following key findings emerged:

### Key Recommendations for Elpis Phase 2

1. **Primary Protocol**: **Server-Sent Events (SSE)** is recommended for LLM streaming
   - Simpler implementation than WebSockets
   - Built-in automatic reconnection
   - Standard HTTP/HTTPS compatibility (firewall-friendly)
   - Sufficient for unidirectional server-to-client streaming
   - Native browser support via EventSource API

2. **Token Streaming**: **Yes, stream token-by-token** (or small chunks)
   - Significantly improves perceived latency
   - Users finish reading responses sooner
   - Critical for chatbot UX where quick initial responses are desirable
   - Both OpenAI and Anthropic APIs use this approach

3. **Implementation Framework**: **FastAPI + sse-starlette**
   - Excellent asyncio integration
   - Production-ready SSE support
   - Clean API design
   - Active community and documentation

4. **Fallback Option**: **WebSockets** for bidirectional scenarios
   - If client-to-server streaming is needed
   - For multi-agent systems with complex interactions
   - When building collaborative real-time features

---

## 1. Streaming Protocols Comparison

### 1.1 Server-Sent Events (SSE)

**Overview**: SSE provides a simple, HTTP-based protocol for server-to-client streaming using persistent connections.

**Technical Characteristics**:
- **Transport**: Standard HTTP/HTTPS (typically HTTP/1.1 or HTTP/2)
- **Direction**: Unidirectional (server → client)
- **Data Format**: Text-based (typically UTF-8 encoded)
- **Content-Type**: `text/event-stream`
- **Reconnection**: Automatic with configurable retry intervals
- **Browser Support**: Native via EventSource API

**Advantages**:
- Simple to implement - leverages standard HTTP
- Firewall and proxy friendly (uses HTTP/HTTPS)
- Automatic reconnection with event ID resumption
- No special infrastructure required
- Lower complexity than WebSockets
- Works well with HTTP/2 multiplexing

**Disadvantages**:
- Unidirectional only (no client → server streaming)
- Text-based (no binary data support)
- EventSource API only supports GET requests (requires workarounds for POST)
- Limited browser connection limits (6 per domain in HTTP/1.1, but HTTP/2 multiplexing helps)

**Best Use Cases**:
- Live notifications and feeds
- Real-time dashboards
- Stock tickers and price updates
- **LLM response streaming** (primary use case for Elpis)
- News feeds and social media updates

### 1.2 WebSocket

**Overview**: WebSocket provides full-duplex bidirectional communication over a single TCP connection.

**Technical Characteristics**:
- **Transport**: WebSocket protocol (ws:// or wss://)
- **Direction**: Bidirectional (client ↔ server)
- **Data Format**: Text and binary
- **Upgrade**: Starts as HTTP, upgrades to WebSocket
- **Reconnection**: Manual (must implement custom logic)

**Advantages**:
- Full bidirectional communication
- Lower latency for back-and-forth exchanges
- Binary data support
- Single persistent connection
- Efficient for high-frequency messaging

**Disadvantages**:
- More complex implementation
- No automatic reconnection (must implement manually)
- May have issues with corporate firewalls/proxies
- Requires special infrastructure considerations
- Higher overhead for simple server-to-client streaming

**Best Use Cases**:
- Chat applications
- Multiplayer games
- Collaborative editing
- Real-time trading platforms
- Bidirectional streaming scenarios

### 1.3 HTTP/2 Streaming

**Overview**: HTTP/2 provides multiplexed streams over a single TCP connection with server push capabilities.

**Technical Characteristics**:
- **Transport**: HTTP/2
- **Direction**: Request/response with server push
- **Multiplexing**: Multiple concurrent streams per connection
- **Performance**: Significant improvements over HTTP/1.1

**Key Performance Metrics**:
- HTTP/2 with 256 concurrent streams: **115,000 requests/second**
- Compared to HTTP/1.1: **6,500 requests/second**
- Best case (4 clients, 256 streams): **225,000 requests/second**
- GET request turnaround time improvement: **~28% at p95 and p99**

**Advantages**:
- Excellent performance with multiplexing
- Single connection per domain
- Header compression (HPACK)
- Server push capability
- Stream prioritization

**Disadvantages**:
- More complex than HTTP/1.1
- Not all proxies/CDNs support server push
- Requires careful configuration
- SSE already works well over HTTP/2

**Best Use Cases**:
- Modern web applications
- API gateways
- Microservices communication
- Works well as transport for SSE

### 1.4 gRPC Streaming

**Overview**: gRPC is a high-performance RPC framework using HTTP/2 and Protocol Buffers.

**Technical Characteristics**:
- **Transport**: HTTP/2
- **Direction**: Unary, server streaming, client streaming, bidirectional
- **Data Format**: Protocol Buffers (binary)
- **Code Generation**: Multi-language support

**Advantages**:
- High performance
- Strong typing with protobuf
- Built-in streaming support (all four types)
- Efficient binary serialization
- Multi-language client generation

**Disadvantages**:
- More complex setup
- Not browser-native (requires grpc-web)
- Binary format (less debuggable)
- **Python-specific**: Streaming RPCs are slower than unary in Python due to threading overhead
- Overkill for simple LLM streaming

**Performance Considerations**:
- gRPC streaming in Python creates extra threads for receiving/sending messages
- This makes streaming RPCs "much slower than unary RPCs in gRPC Python, unlike other languages"
- asyncio can improve performance
- Experimental single-threaded unary-stream implementation can save up to 7% latency per message

**Best Use Cases**:
- Microservices communication
- Internal service-to-service APIs
- Low-latency RPC systems
- Polyglot environments requiring multi-language support

### 1.5 Comparison Matrix

| Feature | SSE | WebSocket | HTTP/2 | gRPC |
|---------|-----|-----------|--------|------|
| **Direction** | Server → Client | Bidirectional | Request/Response + Push | Bidirectional |
| **Transport** | HTTP/HTTPS | WebSocket (ws/wss) | HTTP/2 | HTTP/2 |
| **Data Format** | Text | Text/Binary | Text/Binary | Binary (protobuf) |
| **Complexity** | Low | Medium | Medium | High |
| **Auto Reconnect** | Yes | No | N/A | No |
| **Browser Native** | Yes (EventSource) | Yes | Yes | No (needs grpc-web) |
| **Firewall Friendly** | Yes | Sometimes | Yes | Yes |
| **LLM Streaming** | Excellent | Good | Good (as SSE transport) | Overkill |
| **Setup Effort** | Low | Medium | Medium | High |
| **Python Performance** | Good | Good | Excellent | Fair (threading issues) |

---

## 2. LLM Token Streaming Best Practices

### 2.1 Should Responses Stream Token-by-Token?

**Recommendation**: **YES** - Stream tokens as they are generated (token-by-token or small chunks).

**Rationale**:

1. **User Experience**: "Streaming allows partial LLM outputs to be streamed back as chunks of tokens generated incrementally, which is important for chatbot applications where quick initial responses are desirable"

2. **Perceived Latency**: While streaming has psychological effects, it "genuinely reduces overall latency when considering the complete app-user system, as users finish reading responses sooner"

3. **Industry Standard**: Both OpenAI and Anthropic APIs stream token-by-token by default

4. **Progressive Disclosure**: Users can start reading while the model is still generating

### 2.2 How OpenAI and Anthropic Handle Streaming

**Common Approach**: Both APIs use Server-Sent Events (SSE)

**OpenAI Streaming**:
- Content-Type: `text/event-stream`
- Blocks separated by `\r\n\r\n`
- Each block contains: `data: JSON`
- Usage information: Set `"stream_options": {"include_usage": true}` to get token counts in final message
- Usage provided in last chunk

**Anthropic Streaming**:
- Content-Type: `text/event-stream`
- Similar to OpenAI but includes `event:` line with event type
- Usage information in initial message structure, merged at end
- Can mix content and tool calling in stream

**Example API Call**:
```python
response = completion(
    model="claude-opus-4-20250514",
    messages=messages,
    stream=True
)
```

**Browser Limitation**: These APIs can't be directly consumed using browser EventSource API because:
- EventSource only works for GET requests
- LLM APIs use POST requests
- Must implement custom streaming client or use server-side proxy

### 2.3 Buffering Strategies

#### TokenFlow Buffer Strategy
"TokenFlow frames streaming like video delivery, using a token buffer that generates slightly faster than users consume, then dynamically preempts which requests run to prevent individual buffers from running dry during bursts."

**Performance**:
- Up to **82.5% higher effective throughput**
- Up to **80.2% lower P99 TTFT** (Time To First Token)

#### Chunked Prefilling
"Chunked prefilling splits long prompts into chunks and intermixes them with decode steps for other requests, so ongoing decodes aren't completely halted by new prefills."

**Benefits**:
- Slightly increases TTFT for new requests
- Significantly reduces interruption for ongoing requests
- Better multi-request handling

#### Sarathi-Serve Approach
"Sarathi uses chunked-prefills that compute a prompt's prefill phase over multiple iterations with subsets of tokens, and decode-maximal batching that coalesces ongoing decodes with prefill chunks."

### 2.4 Key Latency Metrics (2026)

1. **First Token Latency (TTFT)**: Time for model to start generating the first token
2. **Per Token Latency**: Time to generate each subsequent token
3. **Total Latency**: End-to-end time for complete response

**Optimization Focus**: Minimize TTFT for better UX, as users can start reading immediately.

---

## 3. Implementation in Python

### 3.1 FastAPI + Server-Sent Events

**Recommended Library**: `sse-starlette`

**Installation**:
```bash
pip install fastapi sse-starlette uvicorn
```

#### Example 1: Basic SSE Streaming

```python
import asyncio
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

app = FastAPI()

async def event_generator(text: str):
    """Stream text character by character"""
    for char in text:
        yield {
            "event": "message",
            "data": char
        }
        await asyncio.sleep(0.02)  # Simulate token generation delay

@app.get("/stream")
async def sse_endpoint(text: str):
    return EventSourceResponse(event_generator(text))
```

#### Example 2: LLM Streaming with LangChain

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
import json

app = FastAPI()

async def send_completion_events(messages, chat):
    """Stream LLM completion events"""
    async for patch in chat.astream_log(messages):
        for op in patch.ops:
            if op["op"] == "add" and op["path"] == "/streamed_output/-":
                # Extract content from the patch operation
                content = op["value"] if isinstance(op["value"], str) else op["value"].content

                # Format as SSE event
                json_dict = {
                    "type": "llm_chunk",
                    "content": content
                }
                yield f"data: {json.dumps(json_dict)}\n\n"

@app.post("/api/completion")
async def stream_completion(messages: list):
    chat = ChatOpenAI(streaming=True)
    return StreamingResponse(
        send_completion_events(messages, chat=chat),
        media_type="text/event-stream"
    )
```

#### Example 3: OpenAI Direct Streaming

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import openai
import json

app = FastAPI()

async def openai_stream(prompt: str):
    """Stream OpenAI responses"""
    client = openai.AsyncOpenAI()

    stream = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True}  # Get token usage
    )

    async for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                yield f"data: {json.dumps({'content': delta.content})}\n\n"

        # Final chunk with usage information
        if hasattr(chunk, 'usage') and chunk.usage:
            yield f"data: {json.dumps({'usage': chunk.usage.model_dump()})}\n\n"

@app.get("/chat")
async def chat(prompt: str):
    return StreamingResponse(
        openai_stream(prompt),
        media_type="text/event-stream"
    )
```

#### Example 4: Custom Generator with Error Handling

```python
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import asyncio
from typing import AsyncGenerator

app = FastAPI()

async def generate_stream() -> AsyncGenerator[dict, None]:
    """Generator with error handling and keep-alive"""
    try:
        for i in range(100):
            # Simulate LLM token generation
            await asyncio.sleep(0.1)

            yield {
                "event": "message",
                "data": f"Token {i}",
                "id": str(i)  # Event ID for resumption
            }

            # Send keep-alive comments every few messages
            if i % 10 == 0:
                yield {"comment": "keep-alive"}

    except Exception as e:
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)})
        }

@app.get("/stream")
async def stream():
    return EventSourceResponse(generate_stream())
```

### 3.2 FastAPI + WebSocket

**Use When**: You need bidirectional communication or client-to-server streaming.

#### Example 1: Basic WebSocket Echo

```python
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            # Process and send back
            response = f"Echo: {data}"
            await websocket.send_text(response)

    except WebSocketDisconnect:
        print("Client disconnected")
```

#### Example 2: Broadcasting to Multiple Clients

```python
from fastapi import FastAPI, WebSocket
from typing import List

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        """Broadcast to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                # Handle disconnected clients
                self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            # Broadcast to all clients
            await manager.broadcast(f"Client {client_id}: {data}")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client {client_id} left")
```

#### Example 3: WebSocket with Heartbeat

```python
from fastapi import FastAPI, WebSocket
import asyncio

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def send_heartbeat():
        """Send periodic heartbeat to keep connection alive"""
        while True:
            try:
                await websocket.send_json({"type": "heartbeat"})
                await asyncio.sleep(30)  # Every 30 seconds
            except Exception:
                break

    # Start heartbeat task
    heartbeat_task = asyncio.create_task(send_heartbeat())

    try:
        while True:
            data = await websocket.receive_json()
            # Process message
            await websocket.send_json({"echo": data})

    except WebSocketDisconnect:
        heartbeat_task.cancel()
```

### 3.3 AsyncIO Patterns for Streaming

#### Pattern 1: Queue-Based Streaming

```python
import asyncio
from asyncio import Queue

async def producer(queue: Queue):
    """Produce tokens to queue"""
    for i in range(100):
        await queue.put(f"token_{i}")
        await asyncio.sleep(0.1)
    await queue.put(None)  # Sentinel value

async def consumer(queue: Queue):
    """Consume tokens from queue"""
    while True:
        token = await queue.get()
        if token is None:
            break
        yield token

async def stream_handler():
    queue = Queue(maxsize=10)  # Bounded queue for backpressure

    # Start producer
    producer_task = asyncio.create_task(producer(queue))

    # Consume and yield
    async for token in consumer(queue):
        yield token

    await producer_task
```

#### Pattern 2: AsyncIO Stream

```python
import asyncio

async def create_stream():
    """Create an asyncio stream"""
    reader, writer = await asyncio.open_connection('api.example.com', 443, ssl=True)

    # Write request
    writer.write(b'GET /stream HTTP/1.1\r\n\r\n')
    await writer.drain()

    # Read response stream
    while True:
        data = await reader.readline()
        if not data:
            break
        yield data.decode('utf-8')

    writer.close()
    await writer.wait_closed()
```

#### Pattern 3: Concurrent Streaming

```python
import asyncio

async def process_stream(stream_id: int):
    """Process individual stream"""
    for i in range(10):
        await asyncio.sleep(0.1)
        yield f"Stream {stream_id}: Token {i}"

async def merge_streams(*streams):
    """Merge multiple streams concurrently"""
    tasks = [asyncio.create_task(stream) for stream in streams]

    # Process all streams concurrently
    for task in asyncio.as_completed(tasks):
        result = await task
        yield result

# Usage
async def main():
    streams = [process_stream(i) for i in range(5)]
    async for item in merge_streams(*streams):
        print(item)
```

---

## 4. Stream of Consciousness Pattern

### 4.1 Continuous Output from LLM

**Concept**: The LLM continuously generates output that multiple clients can subscribe to in real-time.

**Architecture**:
```
┌─────────────┐
│     LLM     │
│  Generator  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Message   │
│   Broker    │
└──────┬──────┘
       │
       ├──────┬──────┬──────┐
       ▼      ▼      ▼      ▼
    Client Client Client Client
      1      2      3      4
```

#### Implementation with FastAPI + SSE

```python
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import asyncio
from asyncio import Queue
from typing import Set

app = FastAPI()

class StreamBroker:
    def __init__(self):
        self.subscribers: Set[Queue] = set()
        self.generator_running = False

    async def subscribe(self) -> Queue:
        """Subscribe to the stream"""
        queue = Queue(maxsize=100)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: Queue):
        """Unsubscribe from the stream"""
        self.subscribers.discard(queue)

    async def publish(self, message: dict):
        """Publish message to all subscribers"""
        # Remove disconnected subscribers
        dead_queues = []

        for queue in self.subscribers:
            try:
                # Non-blocking put with timeout
                await asyncio.wait_for(queue.put(message), timeout=0.1)
            except asyncio.TimeoutError:
                # Queue is full - subscriber is too slow
                dead_queues.append(queue)
            except Exception:
                dead_queues.append(queue)

        # Clean up dead subscribers
        for queue in dead_queues:
            self.unsubscribe(queue)

    async def generate_stream(self):
        """Continuous LLM generation"""
        if self.generator_running:
            return

        self.generator_running = True

        try:
            counter = 0
            while True:
                # Simulate LLM token generation
                await asyncio.sleep(0.1)

                message = {
                    "type": "token",
                    "content": f"Token_{counter}",
                    "id": counter
                }

                await self.publish(message)
                counter += 1

                # Stop if no subscribers
                if not self.subscribers:
                    break

        finally:
            self.generator_running = False

broker = StreamBroker()

@app.get("/stream")
async def stream():
    """Subscribe to the continuous stream"""
    queue = await broker.subscribe()

    # Start generator if not running
    if not broker.generator_running:
        asyncio.create_task(broker.generate_stream())

    async def event_generator():
        try:
            while True:
                message = await queue.get()
                yield {
                    "event": message["type"],
                    "data": message["content"],
                    "id": str(message["id"])
                }
        finally:
            broker.unsubscribe(queue)

    return EventSourceResponse(event_generator())
```

### 4.2 Multiple Client Subscriptions

**Challenge**: How to efficiently broadcast to many clients without blocking.

**Solution**: Use the `websockets` library's `broadcast()` function, which avoids backpressure.

From the websockets documentation:
> "The built-in broadcast() function is similar to the naive way. The main difference is that it doesn't apply backpressure. This provides the best performance by avoiding the overhead of scheduling and running one task per client. broadcast() pushes the message synchronously to all connections even if their write buffers are overflowing. There's no backpressure."

**Implementation**:
```python
from websockets.server import broadcast
import websockets
import asyncio

connected_clients = set()

async def handler(websocket):
    """Handle WebSocket connection"""
    connected_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)

async def broadcaster():
    """Broadcast to all clients"""
    counter = 0
    while True:
        await asyncio.sleep(0.1)
        message = f"Token_{counter}"

        # Broadcast without backpressure
        broadcast(connected_clients, message)
        counter += 1

async def main():
    # Start broadcaster
    asyncio.create_task(broadcaster())

    # Start WebSocket server
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # Run forever

asyncio.run(main())
```

### 4.3 Backpressure Handling

**The Problem**: What happens when clients can't keep up with the stream?

**Key Insight**: "Backpressure, usually a good practice, doesn't work well when broadcasting a message to thousands of clients."

**Strategies**:

1. **No Backpressure (Recommended for Broadcasting)**:
   - Use `websockets.broadcast()` which doesn't apply backpressure
   - Let slow clients' write buffers overflow
   - Rely on ping_timeout to disconnect extremely slow clients

2. **Disconnect Slow Clients**:
   ```python
   # websockets automatically disconnects clients that reach ping_timeout
   # Default ping_timeout will handle clients that get too far behind
   ```

3. **Queue with Bounded Size (SSE)**:
   ```python
   queue = Queue(maxsize=100)

   try:
       await asyncio.wait_for(queue.put(message), timeout=0.1)
   except asyncio.TimeoutError:
       # Queue full - disconnect slow client
       disconnect_client()
   ```

4. **Adaptive Strategies** (Advanced):
   ```python
   async def smart_send(websocket, message):
       """Send with adaptive strategy based on buffer size"""
       buffer_size = websocket.transport.get_write_buffer_size()
       queue_size = message_queue.qsize()

       if buffer_size > 1_000_000:  # 1MB
           # Client very slow - disconnect
           await websocket.close()
       elif buffer_size > 100_000:  # 100KB
           # Client slow - skip some messages or batch
           pass  # Skip this message
       else:
           # Client keeping up - send normally
           await websocket.send(message)
   ```

### 4.4 Connection Management

**Best Practices**:

1. **Track Connections**:
   ```python
   class ConnectionManager:
       def __init__(self):
           self.connections: Set[WebSocket] = set()
           self.connection_metadata: Dict[WebSocket, dict] = {}

       async def connect(self, websocket: WebSocket, metadata: dict = None):
           await websocket.accept()
           self.connections.add(websocket)
           self.connection_metadata[websocket] = metadata or {}

       def disconnect(self, websocket: WebSocket):
           self.connections.discard(websocket)
           self.connection_metadata.pop(websocket, None)
   ```

2. **Heartbeat/Keep-Alive**:
   ```python
   async def heartbeat(websocket: WebSocket):
       """Send periodic heartbeat"""
       while True:
           try:
               await asyncio.sleep(30)
               await websocket.send_json({"type": "ping"})
           except Exception:
               break
   ```

3. **Graceful Shutdown**:
   ```python
   async def shutdown(manager: ConnectionManager):
       """Gracefully close all connections"""
       for websocket in list(manager.connections):
           try:
               await websocket.send_json({"type": "shutdown"})
               await websocket.close()
           except Exception:
               pass
   ```

4. **Connection Limits**:
   ```python
   MAX_CONNECTIONS = 10000

   async def connect_with_limit(websocket: WebSocket, manager: ConnectionManager):
       if len(manager.connections) >= MAX_CONNECTIONS:
           await websocket.close(code=1008, reason="Server at capacity")
           return False

       await manager.connect(websocket)
       return True
   ```

---

## 5. Performance and Reliability

### 5.1 Latency Considerations

**Key Metrics**:
- First Token Latency (TTFT): Time to first token
- Per Token Latency: Time per subsequent token
- Total Latency: End-to-end time

**Optimization Strategies**:

1. **Minimize TTFT**:
   - Use streaming to start delivering immediately
   - Optimize prompt preprocessing
   - Use prompt caching (Anthropic/OpenAI)
     - "Cached input tokens are 10x cheaper in dollars per token"
     - "Anthropic claims prompt caching can reduce latency by up to 85% for long prompts"

2. **Buffer Management**:
   - Small buffers for lower latency
   - Larger buffers for stability
   - TokenFlow: Dynamic buffer management

3. **Network Optimization**:
   - Use HTTP/2 when possible
   - Enable compression
   - Minimize round trips

### 5.2 Error Handling and Reconnection

#### SSE Automatic Reconnection

**Browser Behavior**:
- EventSource automatically reconnects when connection drops
- Default retry interval: 3 seconds
- Server can specify custom retry: `retry: 5000` (5 seconds in milliseconds)

**Last Event ID Resumption**:
```python
# Server side
async def sse_generator(last_event_id: str = None):
    start_id = int(last_event_id) if last_event_id else 0

    for i in range(start_id, 1000):
        yield {
            "id": str(i),  # Critical: Include ID for resumption
            "event": "message",
            "data": f"Event {i}"
        }
```

**Client Side (Browser)**:
```javascript
const eventSource = new EventSource('/stream');

eventSource.onmessage = (event) => {
    console.log('Data:', event.data);
    console.log('Last ID:', event.lastEventId);  // Automatically tracked
};

eventSource.onerror = (error) => {
    console.log('Error - will auto-reconnect');
    // Browser sends Last-Event-ID header on reconnection
};
```

**Server Receives Last-Event-ID**:
```python
from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse

@app.get("/stream")
async def stream(request: Request):
    last_event_id = request.headers.get("Last-Event-ID")
    return EventSourceResponse(sse_generator(last_event_id))
```

#### WebSocket Reconnection

**Manual Implementation Required**:
```javascript
class ReconnectingWebSocket {
    constructor(url, options = {}) {
        this.url = url;
        this.reconnectDelay = options.reconnectDelay || 1000;
        this.maxReconnectDelay = options.maxReconnectDelay || 30000;
        this.reconnectAttempts = 0;
        this.connect();
    }

    connect() {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            this.reconnectAttempts = 0;
            this.reconnectDelay = 1000;
        };

        this.ws.onclose = () => {
            this.reconnect();
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    reconnect() {
        this.reconnectAttempts++;
        const delay = Math.min(
            this.reconnectDelay * Math.pow(2, this.reconnectAttempts),
            this.maxReconnectDelay
        );

        setTimeout(() => this.connect(), delay);
    }
}
```

**Python Server Side**:
```python
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=60.0  # 60 second timeout
                )
                await websocket.send_text(f"Echo: {data}")

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()
```

### 5.3 Message Ordering Guarantees

**SSE**:
- Messages delivered in order (HTTP is ordered)
- Single TCP connection maintains order
- Event IDs help track sequence

**WebSocket**:
- Messages delivered in order over single connection
- No built-in sequence numbers (must implement if needed)

**HTTP/2**:
- Individual streams are ordered
- Multiplexed streams may interleave
- Stream IDs maintain logical ordering

### 5.4 Resource Usage

#### Memory Usage (WebSocket)

From the `websockets` library documentation:

**Per-Connection Memory**: 70 KiB per connection with default settings

**Memory Formula**: `4 * max_size * max_queue` bytes
- Default max_size: 1 MiB
- Default max_queue: 32
- Default memory: ~128 MB per connection (worst case)

**Production Tuning**:
```python
import websockets

async def handler(websocket):
    # Custom limits for production
    websocket.max_size = 256 * 1024  # 256 KiB
    websocket.max_queue = 16
    # Memory per connection: 4 * 256KB * 16 = ~16 MB worst case
```

**Configuration Parameters**:
```python
import websockets

server = await websockets.serve(
    handler,
    "localhost",
    8765,
    max_size=256 * 1024,      # 256 KiB max message size
    max_queue=16,              # Max queued messages
    read_limit=64 * 1024,      # 64 KiB read buffer
    write_limit=64 * 1024,     # 64 KiB write buffer
    compression=None           # Disable compression to save memory
)
```

**Scaling Considerations**:
- 1,000 connections: ~70 MB baseline
- 10,000 connections: ~700 MB baseline
- With high message rates, actual usage can be much higher
- "When there are lots of connections with high message rates, create_task might keep the number of event loop's coroutines increasing, leading to unwanted memory usage"

**Memory Optimization**:
```python
# Compress once for broadcast (saves RAM)
import zlib

compressed_message = zlib.compress(message.encode())

# Broadcast compressed message
for connection in connections:
    await connection.send(compressed_message)
```

#### Connection Limits

**System Limits**:
- OS file descriptor limits
- Network bandwidth
- Memory constraints
- CPU for connection handling

**Best Practices**:
```python
# Limit concurrent connections
import asyncio

MAX_CONNECTIONS = 10000
connection_semaphore = asyncio.Semaphore(MAX_CONNECTIONS)

async def limited_handler(websocket):
    async with connection_semaphore:
        await handler(websocket)
```

#### SSE Resource Usage

**Advantages over WebSocket**:
- Simpler protocol = less overhead
- HTTP/2 multiplexing shares connection resources
- No per-connection WebSocket upgrade overhead

**Considerations**:
- One connection per client (like WebSocket)
- HTTP/1.1: Limited to 6 connections per domain in browsers
- HTTP/2: Multiplexed streams, much better scalability

### 5.5 Rate Limiting

**Implementation**:
```python
from fastapi import FastAPI, HTTPException, Request
from time import time
from collections import defaultdict

app = FastAPI()

# Simple rate limiter
rate_limits = defaultdict(lambda: {"count": 0, "reset": time() + 60})

def check_rate_limit(client_id: str, max_requests: int = 100) -> bool:
    """Check if client is within rate limit"""
    current_time = time()

    if current_time > rate_limits[client_id]["reset"]:
        rate_limits[client_id] = {"count": 0, "reset": current_time + 60}

    if rate_limits[client_id]["count"] >= max_requests:
        return False

    rate_limits[client_id]["count"] += 1
    return True

@app.get("/stream")
async def stream(request: Request):
    client_id = request.client.host

    if not check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Return stream...
```

**LLM API Rate Limits**:
From research: "When a rate limit error occurs (HTTP 429 from OpenAI or Anthropic), your client should not immediately retry; instead, implement exponential backoff for retries."

**Exponential Backoff**:
```python
import asyncio
import random

async def retry_with_backoff(func, max_retries=5):
    """Retry with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return await func()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise

            # Exponential backoff with jitter
            delay = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)
```

---

## 6. Recommendations for Elpis Phase 2

### 6.1 Primary Recommendation: SSE with FastAPI

**Rationale**:
1. **Simplicity**: SSE is simpler to implement and maintain than WebSocket
2. **Sufficient**: One-way streaming (server → client) meets LLM streaming requirements
3. **Reliability**: Built-in automatic reconnection with event ID resumption
4. **Compatibility**: Works over standard HTTP/HTTPS, firewall-friendly
5. **Industry Standard**: OpenAI and Anthropic both use SSE for streaming
6. **Performance**: Adequate for LLM token streaming, works well with HTTP/2

**Implementation Stack**:
- **Framework**: FastAPI
- **SSE Library**: sse-starlette
- **Async Runtime**: asyncio
- **Web Server**: Uvicorn

**Sample Architecture**:
```
┌─────────────────┐
│   FastAPI App   │
│   + sse-starlette│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Provider   │
│ (OpenAI/Anthropic)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SSE Stream     │
│  to Clients     │
└─────────────────┘
```

### 6.2 When to Use WebSocket

**Use WebSocket if**:
1. You need bidirectional streaming (client also sends continuous data)
2. You're building multi-agent systems with agent-to-agent communication
3. You need binary data streaming
4. You're building collaborative features (multiple users editing together)

**Not Needed For**:
- Simple LLM request/response with streaming
- Server-to-client notifications
- Progress updates and status streaming

### 6.3 Implementation Roadmap

**Phase 1: Basic SSE Streaming**
```python
# 1. Install dependencies
# pip install fastapi sse-starlette uvicorn

# 2. Create basic SSE endpoint
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import asyncio

app = FastAPI()

@app.get("/stream")
async def stream_response(prompt: str):
    async def generate():
        # Simulate LLM streaming
        response = await call_llm(prompt, stream=True)
        async for token in response:
            yield {
                "event": "token",
                "data": token
            }

    return EventSourceResponse(generate())
```

**Phase 2: Add Error Handling & Reconnection**
```python
async def generate_with_recovery(last_event_id: str = None):
    try:
        start_index = int(last_event_id) if last_event_id else 0

        async for i, token in enumerate(llm_stream(), start=start_index):
            yield {
                "id": str(i),  # For resumption
                "event": "token",
                "data": token
            }

    except Exception as e:
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)})
        }
```

**Phase 3: Add Broadcasting & Multiple Clients**
```python
# Use StreamBroker pattern from Section 4.1
class StreamBroker:
    # Implementation from earlier...
    pass

broker = StreamBroker()

@app.get("/stream/{conversation_id}")
async def stream(conversation_id: str):
    queue = await broker.subscribe(conversation_id)

    async def event_generator():
        try:
            while True:
                message = await queue.get()
                yield message
        finally:
            broker.unsubscribe(conversation_id, queue)

    return EventSourceResponse(event_generator())
```

**Phase 4: Production Hardening**
- Add rate limiting
- Implement monitoring and metrics
- Add connection limits
- Configure proper timeouts
- Add health checks
- Implement graceful shutdown

### 6.4 Testing Strategy

**Unit Tests**:
```python
from fastapi.testclient import TestClient
import pytest

def test_sse_stream():
    client = TestClient(app)

    with client.stream("GET", "/stream?prompt=test") as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"

        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(line[6:])

        assert len(events) > 0
```

**Integration Tests**:
```python
import asyncio
import httpx

async def test_sse_integration():
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", "http://localhost:8000/stream") as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    print(f"Received: {line[6:]}")
```

**Load Tests**:
```python
import asyncio
import aiohttp

async def load_test(num_clients=100):
    """Simulate multiple concurrent clients"""
    async def client():
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/stream') as resp:
                async for line in resp.content:
                    pass  # Process stream

    tasks = [asyncio.create_task(client()) for _ in range(num_clients)]
    await asyncio.gather(*tasks)
```

### 6.5 Monitoring and Observability

**Key Metrics to Track**:
1. Active connection count
2. Messages per second
3. Average latency (TTFT and per-token)
4. Error rates
5. Reconnection rates
6. Memory usage per connection

**Implementation**:
```python
from prometheus_client import Counter, Gauge, Histogram
import time

# Metrics
active_connections = Gauge('sse_active_connections', 'Number of active SSE connections')
messages_sent = Counter('sse_messages_sent', 'Total messages sent')
latency = Histogram('sse_message_latency', 'Message latency in seconds')

class MonitoredStreamBroker(StreamBroker):
    async def subscribe(self):
        queue = await super().subscribe()
        active_connections.inc()
        return queue

    def unsubscribe(self, queue):
        super().unsubscribe(queue)
        active_connections.dec()

    async def publish(self, message):
        start = time.time()
        await super().publish(message)
        latency.observe(time.time() - start)
        messages_sent.inc()
```

---

## 7. Code Examples Summary

### Quick Reference: FastAPI SSE

**Minimal Example**:
```python
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import asyncio

app = FastAPI()

@app.get("/stream")
async def stream():
    async def generate():
        for i in range(10):
            await asyncio.sleep(0.5)
            yield {"data": f"Message {i}"}

    return EventSourceResponse(generate())
```

**Production Example**:
```python
from fastapi import FastAPI, Request, HTTPException
from sse_starlette.sse import EventSourceResponse
import asyncio
import json

app = FastAPI()

@app.get("/stream")
async def stream(request: Request, prompt: str):
    # Get last event ID for resumption
    last_event_id = request.headers.get("Last-Event-ID")

    async def generate():
        try:
            start_id = int(last_event_id) if last_event_id else 0

            # Call LLM API with streaming
            async for i, token in enumerate(llm_stream(prompt), start=start_id):
                yield {
                    "id": str(i),
                    "event": "token",
                    "data": json.dumps({"content": token, "index": i})
                }

                # Periodic keep-alive
                if i % 20 == 0:
                    yield {"comment": "keep-alive"}

            # Final event
            yield {
                "event": "done",
                "data": json.dumps({"status": "complete"})
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(generate())
```

### Quick Reference: Client-Side JavaScript

**Basic EventSource**:
```javascript
const eventSource = new EventSource('/stream?prompt=Hello');

eventSource.addEventListener('token', (event) => {
    const data = JSON.parse(event.data);
    console.log('Token:', data.content);
});

eventSource.addEventListener('done', (event) => {
    console.log('Stream complete');
    eventSource.close();
});

eventSource.addEventListener('error', (event) => {
    console.error('Stream error');
    // Browser will auto-reconnect unless you close()
});
```

**Advanced EventSource with POST**:
```javascript
// EventSource only supports GET, so use fetch for POST
async function streamPost(url, data) {
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream'
        },
        body: JSON.stringify(data)
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split('\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                console.log('Received:', data);
            }
        }
    }
}
```

---

## 8. References and Resources

### Official Documentation

**FastAPI**:
- [FastAPI WebSockets Documentation](https://fastapi.tiangolo.com/advanced/websockets/)
- [FastAPI Official Site](https://fastapi.tiangolo.com/)

**SSE Libraries**:
- [sse-starlette PyPI](https://pypi.org/project/sse-starlette/)
- [websockets Documentation](https://websockets.readthedocs.io/)

**Browser APIs**:
- [Server-Sent Events - MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)

**LLM Providers**:
- [OpenAI API Documentation](https://platform.openai.com/docs/guides/latency-optimization)
- [Anthropic API Documentation](https://docs.litellm.ai/docs/providers/anthropic)

### Protocol Comparisons

- [WebSockets vs SSE - Ably Blog](https://ably.com/blog/websockets-vs-sse)
- [WebSockets vs SSE - WebSocket.org](https://websocket.org/comparisons/sse/)
- [SSE vs WebSockets - SoftwareMill](https://softwaremill.com/sse-vs-websockets-comparing-real-time-communication-protocols/)
- [Streaming HTTP vs WebSocket vs SSE](https://dev.to/mechcloud_academy/streaming-http-vs-websocket-vs-sse-a-comparison-for-real-time-data-1geo)
- [Server-Sent Events vs WebSockets - freeCodeCamp](https://www.freecodecamp.org/news/server-sent-events-vs-websockets/)
- [gRPC vs WebSocket - Ably](https://ably.com/topic/grpc-vs-websocket)

### Implementation Guides

**SSE Implementations**:
- [Implementing SSE with FastAPI - Mahdi Jafari](https://mahdijafaridev.medium.com/implementing-server-sent-events-sse-with-fastapi-real-time-updates-made-simple-6492f8bfc154)
- [Server-Sent Events with FastAPI - Nanda Gopal](https://medium.com/@nandagopal05/server-sent-events-with-python-fastapi-f1960e0c8e4b)
- [Real-Time Notifications with SSE - İnan Delibaş](https://medium.com/@inandelibas/real-time-notifications-in-python-using-sse-with-fastapi-1c8c54746eb7)
- [FastAPI SSE Streaming - Clay-Technology World](https://clay-atlas.com/us/blog/2024/11/02/en-python-fastapi-server-sent-events-sse/)

**WebSocket Implementations**:
- [FastAPI WebSockets - Better Stack](https://betterstack.com/community/guides/scaling-python/fastapi-websockets/)
- [Build Real-Time Applications - VideoSDK](https://www.videosdk.live/developer-hub/websocket/fastapi-websocket)
- [FastAPI and WebSockets - UnfoldAI](https://unfoldai.com/fastapi-and-websockets/)

**LLM Streaming Examples**:
- [Building Streaming LLMs with FastAPI - Louis Sanna](https://dev.to/louis-sanna/mastering-real-time-ai-a-developers-guide-to-building-streaming-llms-with-fastapi-and-transformers-2be8)
- [LangChain with FastAPI Streaming](https://dev.to/louis-sanna/integrating-langchain-with-fastapi-for-asynchronous-streaming-5d0o)
- [Real-time OpenAI Streaming - Sevalla](https://sevalla.com/blog/real-time-openai-streaming-fastapi/)
- [LLM Web App with SSE - Zachary Huang](https://medium.com/@zh2408/build-an-llm-web-app-in-python-from-scratch-part-4-fastapi-background-tasks-sse-f588d6c2166b)

### Performance and Scaling

**LLM Streaming**:
- [How Streaming LLM APIs Work - Simon Willison](https://til.simonwillison.net/llms/streaming-llm-apis)
- [Streaming Response Structures - Sirsh Amarteifio](https://medium.com/percolation-labs/comparing-the-streaming-response-structure-for-different-llm-apis-2b8645028b41)
- [LLM Streaming Latency Optimization](https://medium.com/@QuarkAndCode/llm-streaming-latency-cut-ttft-smooth-tokens-fix-cold-starts-f2be60d26b89)
- [LLM Latency Benchmark 2026](https://research.aimultiple.com/llm-latency-benchmark/)

**WebSocket Performance**:
- [Broadcasting Messages - websockets docs](https://websockets.readthedocs.io/en/stable/topics/broadcast.html)
- [Memory and Buffers - websockets docs](https://websockets.readthedocs.io/en/stable/topics/memory.html)
- [Scaling WebSockets - Ably](https://ably.com/topic/the-challenge-of-scaling-websockets)

**HTTP/2 Performance**:
- [HTTP/2 Multiplexing Performance](https://dev.to/sibiraj/understanding-http2-parallel-requests-streams-vs-connections-3anf)
- [Improve Throughput with HTTP/2 - Vespa Blog](https://blog.vespa.ai/http2/)
- [HTTP/2 Multiplexing Deep Dive](https://blog.codavel.com/http2-multiplexing)

### Reliability and Best Practices

**SSE Reliability**:
- [SSE Connection Resilience - Ithy](https://ithy.com/article/sse-connection-resilience-qwo3x8pb)
- [Ensuring Reliable Streaming - Ithy](https://ithy.com/article/sse-streaming-retries-v0p7rdp1)
- [SSE Practical Guide - Tiger Abrodi](https://tigerabrodi.blog/server-sent-events-a-practical-guide-for-the-real-world)

**Production Considerations**:
- [Prompt Caching - ngrok blog](https://ngrok.com/blog/prompt-caching/)
- [Rate Limits for LLM Providers - Requesty](https://www.requesty.ai/blog/rate-limits-for-llm-providers-openai-anthropic-and-deepseek)
- [gRPC Performance Best Practices](https://grpc.io/docs/guides/performance/)

---

## 9. Conclusion

For Elpis Phase 2, **Server-Sent Events (SSE) with FastAPI** is the recommended approach for LLM response streaming. This choice is based on:

1. **Industry Alignment**: Both OpenAI and Anthropic use SSE for streaming
2. **Simplicity**: Easier to implement and maintain than WebSocket
3. **Reliability**: Built-in automatic reconnection with event resumption
4. **Performance**: Adequate for LLM token streaming with excellent scalability
5. **Compatibility**: Works over standard HTTP/HTTPS, firewall-friendly
6. **Future-Proof**: Can be enhanced with HTTP/2 multiplexing as needed

The streaming implementation should prioritize:
- Token-by-token streaming for optimal UX
- Proper error handling and reconnection
- Event IDs for stream resumption
- Rate limiting and connection management
- Monitoring and observability

WebSocket should be reserved for future features requiring bidirectional communication, such as multi-agent systems or collaborative features.

---

**Research Completed**: 2026-01-12
**Researcher**: Streaming Researcher
**Status**: Ready for Implementation
