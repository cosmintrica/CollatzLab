const DEFAULT_TIMEOUT_MS = 15_000;

function fetchWithTimeout(url, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(timer));
}

export async function readJson(url) {
  const response = await fetchWithTimeout(url);
  if (!response.ok) {
    throw new Error(`Request failed for ${url}`);
  }
  return response.json();
}

export async function readOptionalJson(url) {
  try {
    const response = await fetchWithTimeout(url);
    if (!response.ok) {
      return null;
    }
    return response.json();
  } catch {
    return null;
  }
}

export async function postJson(url, payload) {
  const options = {
    method: "POST",
    headers: {}
  };
  if (payload !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(payload);
  }
  const response = await fetchWithTimeout(url, options, 30_000);
  if (!response.ok) {
    let message = `Request failed for ${url}`;
    try {
      const body = await response.json();
      message = body.detail ?? body.message ?? message;
    } catch {
      // Ignore JSON parse failures and fall back to the generic message.
    }
    throw new Error(message);
  }
  return response.json();
}

export async function deleteJson(url) {
  const response = await fetchWithTimeout(url, { method: "DELETE" });
  if (!response.ok) {
    let message = `Request failed for ${url}`;
    try {
      const body = await response.json();
      message = body.detail ?? body.message ?? message;
    } catch {
      // Ignore JSON parse failures.
    }
    throw new Error(message);
  }
  return response.json();
}
