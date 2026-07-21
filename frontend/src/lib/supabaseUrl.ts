// Normalizes a Supabase project URL in case it was copied with a REST path
// already attached (e.g. "https://xxxx.supabase.co/rest/v1/") -- every
// Supabase client (service-role, anon browser, anon session-server, proxy)
// appends its own API path, so the base URL must be bare.
export function normalizeSupabaseUrl(url: string): string {
  let clean = url.trim().replace(/\/+$/, "");
  for (const suffix of ["/rest/v1", "/rest", "/auth/v1", "/auth"]) {
    if (clean.endsWith(suffix)) {
      clean = clean.slice(0, -suffix.length);
      break;
    }
  }
  return clean;
}
