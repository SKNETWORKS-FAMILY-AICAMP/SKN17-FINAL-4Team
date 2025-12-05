import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000/api';

const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

const SAFE_METHODS = new Set(['get', 'head', 'options', 'trace']);
let csrfToken: string | null = null;
let csrfPromise: Promise<string> | null = null;

const setCsrfHeader = (config: AxiosRequestConfig) => {
  if (!config.headers) {
    config.headers = {};
  }
  if (csrfToken) {
    config.headers['X-CSRFToken'] = csrfToken;
  }
  return config;
};

async function fetchCsrfToken() {
  if (csrfToken) {
    return csrfToken;
  }
  if (!csrfPromise) {
    csrfPromise = apiClient
      .get('/accounts/csrf/')
      .then((res) => {
        csrfToken = res.data?.csrfToken ?? null;
        if (csrfToken) {
          apiClient.defaults.headers.common['X-CSRFToken'] = csrfToken;
        }
        return csrfToken ?? '';
      })
      .finally(() => {
        csrfPromise = null;
      });
  }
  return csrfPromise;
}

apiClient.interceptors.request.use(async (config) => {
  const method = config.method?.toLowerCase();
  if (method && !SAFE_METHODS.has(method)) {
    await fetchCsrfToken();
    setCsrfHeader(config);
  }
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error?.response?.status === 403 && error.response?.data?.detail === 'CSRF Failed: CSRF token missing or incorrect.') {
      csrfToken = null;
    }
    return Promise.reject(error);
  },
);

export async function initCsrfToken() {
  await fetchCsrfToken();
}

export { apiClient as api };

