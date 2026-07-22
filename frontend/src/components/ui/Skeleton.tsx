import styles from "./Skeleton.module.css";

interface Props {
  width?: string | number;
  height?: string | number;
  radius?: string;
  className?: string;
}

// Shimmering placeholder block for loading states -- replaces "render
// nothing until the fetch resolves," which every page did until now.
export default function Skeleton({ width = "100%", height = "1rem", radius = "var(--radius-sm)", className }: Props) {
  return (
    <div
      className={`${styles.skeleton} ${className ?? ""}`}
      style={{ width, height, borderRadius: radius }}
      aria-hidden="true"
    />
  );
}
