import type { CSSProperties } from "react";
import './character.css'

type CharacterProps = {
  name: string;
  image: string;
  selected: boolean;
  onClick: () => void;
  style?: CSSProperties;
  size?: number;
  imageStyle?: CSSProperties;
};

export function Character({
  name,
  image,
  selected,
  onClick,
  style,
  size,
  imageStyle,
}: CharacterProps) {
  return (
    <button
      type="button"
      className={`character ${selected ? "selected" : ""}`}
      onClick={onClick}
      style={style}
      aria-pressed={selected}
      aria-label={`${name} 선택`}
    >
      <span className="character__glow" />

      <span className="character__float">
        <img
          className="character__image"
          src={image}
          alt={name}
          draggable={false}
          style={{ ...(size ? { width: size, height: size * (200 / 170) } : {}), ...imageStyle }}
        />
      </span>

      <span className="character__name">
        {selected && (
          <span className="character__check" aria-hidden="true">
            ✓
          </span>
        )}
        {name}
      </span>
    </button>
  );
}