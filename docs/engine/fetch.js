/**
 * Streaming download with progress.
 *
 * The body is collected as chunks and concatenated at the end instead of being
 * written into a buffer sized from `content-length`: static hosts and CDNs serve
 * the model gzip/brotli-compressed, so that header is the size *on the wire*
 * while the decoded body is larger — trusting it overflows the buffer
 * ("RangeError: offset is out of bounds").
 */
export async function fetchWithProgress(url, onProgress = () => {}) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`טעינת הקובץ נכשלה (HTTP ${response.status})`);

  const total = Number(response.headers.get("content-length") || 0);
  if (!response.body) {
    const whole = new Uint8Array(await response.arrayBuffer());
    onProgress(1);
    return whole;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    // content-length is a lower bound on compressed responses, so clamp it
    if (total) onProgress(Math.min(1, loaded / total));
  }

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  onProgress(1);
  return bytes;
}
