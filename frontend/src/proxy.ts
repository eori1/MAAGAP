import { createServerClient } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";
import { normalizeSupabaseUrl } from "./lib/supabaseUrl";

// Next.js 16 renamed Middleware to Proxy (same mechanics, new file/export
// name). This refreshes the Supabase session cookie on every request and
// gates unauthenticated users to the login screen (site root, "/") -- an
// "optimistic" check per Next.js's auth guide; role-based authorization
// happens server-side in each page/API route via getSessionProfile().
const LOGIN_PATH = "/";
// The password-recovery link lands here with a one-time code that the
// browser client exchanges for a session client-side; no cookie exists yet
// on this first request, so it must be reachable without an existing session.
const PUBLIC_PATHS = [LOGIN_PATH, "/reset-password"];

export async function proxy(request: NextRequest) {
  let response = NextResponse.next({ request });

  const supabase = createServerClient(
    normalizeSupabaseUrl(process.env.NEXT_PUBLIC_SUPABASE_URL!),
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll();
        },
        setAll(cookiesToSet) {
          for (const { name, value } of cookiesToSet) {
            request.cookies.set(name, value);
          }
          response = NextResponse.next({ request });
          for (const { name, value, options } of cookiesToSet) {
            response.cookies.set(name, value, options);
          }
        },
      },
    },
  );

  const {
    data: { user },
  } = await supabase.auth.getUser();

  const pathname = request.nextUrl.pathname;
  const isPublic = PUBLIC_PATHS.includes(pathname);

  if (!user && !isPublic) {
    const loginUrl = new URL(LOGIN_PATH, request.url);
    loginUrl.searchParams.set("next", pathname);
    return NextResponse.redirect(loginUrl);
  }

  if (user && pathname === LOGIN_PATH) {
    return NextResponse.redirect(new URL("/dashboard", request.url));
  }

  return response;
}

export const config = {
  matcher: [
    "/((?!api|_next/static|_next/image|favicon.ico|.*\\.png$|.*\\.svg$).*)",
  ],
};
