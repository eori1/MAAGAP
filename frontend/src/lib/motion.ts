import type { Transition, Variants } from "framer-motion";

// Mirrors the --duration-*/--ease-standard values in styles/tokens.css so
// CSS-transition components and Framer Motion components move at the same
// rhythm rather than drifting apart over time.
export const DURATION = { fast: 0.12, base: 0.2, slow: 0.32 };
export const EASE_STANDARD: Transition["ease"] = [0.4, 0, 0.2, 1];

export const fadeInUp: Variants = {
  hidden: { opacity: 0, y: 8 },
  visible: { opacity: 1, y: 0, transition: { duration: DURATION.base, ease: EASE_STANDARD } },
};

export const staggerContainer: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.06, delayChildren: 0.04 } },
};

export const drawerSlide: Variants = {
  closed: { x: "-100%" },
  open: { x: 0, transition: { type: "spring", stiffness: 320, damping: 32 } },
};
