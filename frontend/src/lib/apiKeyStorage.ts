'use client';

const STORAGE_KEY = 'recyclevision_openai_api_key';

export function getStoredOpenAIKey(): string {
  if (typeof window === 'undefined') return '';
  return localStorage.getItem(STORAGE_KEY) || '';
}

export function setStoredOpenAIKey(key: string): void {
  if (typeof window === 'undefined') return;
  if (!key || !key.trim()) {
    localStorage.removeItem(STORAGE_KEY);
  } else {
    localStorage.setItem(STORAGE_KEY, key.trim());
  }
  // Dispatch custom event so all open tabs/components re-read the key
  window.dispatchEvent(new Event('recyclevision_key_changed'));
}
