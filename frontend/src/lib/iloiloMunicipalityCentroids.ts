// Approximate centroid coordinates for Iloilo Province municipalities/city,
// derived by averaging the original demo_projects.csv's per-project lat/lng
// (itself municipality-centroid + jitter). The synthetic `projects` table
// only stores a municipality name, not coordinates, so the map view derives
// a stable per-project pin by jittering these centroids deterministically.
const CENTROIDS: Record<string, [number, number]> = {
  "calinog": [11.12006, 122.52755],
  "new lucena": [10.88305, 122.58099],
  "lambunao": [11.05224, 122.48415],
  "concepcion": [11.27966, 123.11416],
  "san joaquin": [10.59357, 122.13919],
  "zarraga": [10.82897, 122.61039],
  "badiangan": [10.99471, 122.56735],
  "dumangas": [10.82919, 122.71067],
  "igbaras": [10.71891, 122.26732],
  "leon": [10.77823, 122.38241],
  "estancia": [11.45526, 123.15233],
  "pototan": [10.95492, 122.62501],
  "tigbauan": [10.67217, 122.37766],
  "alimodian": [10.82567, 122.43102],
  "miagao": [10.63993, 122.23345],
  "sara": [11.26674, 123.01836],
  "san enrique": [11.06123, 122.67095],
  "janiuay": [10.95325, 122.50266],
  "passi city": [11.11786, 122.64661],
  "santa barbara": [10.83576, 122.52673],
  "sta. barbara": [10.83576, 122.52673],
  "guimbal": [10.6653, 122.32446],
  "lemery": [11.2262, 122.90378],
  "tubungan": [10.77724, 122.29687],
  "cabatuan": [10.88523, 122.48477],
  "balasan": [11.46319, 123.06179],
  "banate": [11.03724, 122.79469],
  "barotac nuevo": [10.88623, 122.70936],
  "batad": [11.36459, 123.03125],
  "oton": [10.70496, 122.47142],
  "maasin": [10.8936, 122.43723],
  "san dionisio": [11.30898, 123.0931],
  "barotac viejo": [11.0645, 122.8467],
  "carles": [11.56684, 123.1604],
  "leganes": [10.78744, 122.59724],
  "bingawan": [11.22676, 122.60202],
  "pavia": [10.7775, 122.54057],
  "duenas": [11.06365, 122.62584],
  "dueñas": [11.06365, 122.62584],
  // Not present in the demo CSV sample -- approximate public coordinates.
  "iloilo city": [10.7202, 122.5621],
  "san miguel": [10.75, 122.5765],
  "anilao": [10.7614, 122.4614],
  "ajuy": [11.1152, 123.0248],
  "dingle": [11.0004, 122.6774],
};

const FALLBACK: [number, number] = CENTROIDS["iloilo city"];

function normalize(name: string): string {
  return name.trim().toLowerCase();
}

/** Simple deterministic string hash (FNV-1a), used to jitter a centroid
 * stably per project_id so the same project always renders at the same pin. */
function hashString(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Returns a stable [lat, lng] pin for a project, jittered within ~0.03
 * degrees of its municipality's centroid. */
export function getProjectCoordinates(municipality: string, projectId: string): [number, number] {
  const [lat, lng] = CENTROIDS[normalize(municipality)] ?? FALLBACK;
  const h = hashString(projectId);
  const jitterLat = (((h & 0xffff) / 0xffff) - 0.5) * 0.06;
  const jitterLng = ((((h >>> 16) & 0xffff) / 0xffff) - 0.5) * 0.06;
  return [lat + jitterLat, lng + jitterLng];
}
