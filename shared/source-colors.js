export const SOURCE_PALETTE = [
  { rgb: [0.82, 0.23, 0.2], hsl: "4 72% 44%" },
  { rgb: [0.12, 0.4, 0.74], hsl: "211 74% 40%" },
  { rgb: [0.1, 0.52, 0.4], hsl: "162 68% 30%" },
  { rgb: [0.78, 0.48, 0.08], hsl: "34 82% 40%" },
  { rgb: [0.46, 0.28, 0.7], hsl: "266 50% 44%" },
  { rgb: [0.74, 0.2, 0.46], hsl: "332 60% 42%" },
  { rgb: [0.08, 0.5, 0.6], hsl: "188 76% 34%" },
  { rgb: [0.52, 0.36, 0.16], hsl: "28 54% 34%" },
];

export function sourceColor(id) {
  const index = Math.max(0, Number(id) - 1);
  return SOURCE_PALETTE[index % SOURCE_PALETTE.length];
}
