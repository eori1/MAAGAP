export async function fetchProjects() {
  try {
    const res = await fetch("/api/projects");
    if (!res.ok) throw new Error("Failed to fetch");
    return await res.json();
  } catch (error) {
    console.error("Error fetching projects:", error);
    return [];
  }
}
