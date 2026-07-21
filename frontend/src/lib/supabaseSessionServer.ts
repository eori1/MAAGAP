import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";
import { normalizeSupabaseUrl } from "./supabaseUrl";

// Anon-key, cookie-aware client for Server Components and Route Handlers --
// resolves "who is logged in" from the session cookie (refreshed by
// src/proxy.ts). Distinct from supabaseServer.ts, which uses the
// service-role key for unrestricted backend queries.
export async function getSupabaseSessionServerClient() {
  const cookieStore = await cookies();

  return createServerClient(
    normalizeSupabaseUrl(process.env.NEXT_PUBLIC_SUPABASE_URL!),
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet) {
          try {
            for (const { name, value, options } of cookiesToSet) {
              cookieStore.set(name, value, options);
            }
          } catch {
            // Called from a Server Component render (cookies are read-only
            // there); the proxy is responsible for refreshing the session.
          }
        },
      },
    },
  );
}

export interface SessionProfile {
  userId: string;
  email: string | undefined;
  role: "manager" | "inspector" | "admin";
  inspectorId: string | null;
  fullName: string | null;
}

/** Resolves the current session's user + profile (role, inspector_id), or null if unauthenticated. */
export async function getSessionProfile(): Promise<SessionProfile | null> {
  const supabase = await getSupabaseSessionServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return null;

  const { data: profile } = await supabase
    .from("profiles")
    .select("role, inspector_id, full_name")
    .eq("id", user.id)
    .single();

  return {
    userId: user.id,
    email: user.email,
    role: (profile?.role as SessionProfile["role"]) ?? "inspector",
    inspectorId: profile?.inspector_id ?? null,
    fullName: profile?.full_name ?? null,
  };
}
