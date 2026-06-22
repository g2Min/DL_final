import type { CSSProperties } from "react";

type CharacterData = {
  id: string;
  name: string;
  image: string;
  position: {
    left: string;
    top: string;
  };
  size?: number;
  imageStyle?: CSSProperties;
};

export const characters: CharacterData[] = [
  {
    id: "squirrel",
    name: "다람쥐",
    image: "/static/characters_transparent/squirrel_transparent.png",
    position: {
      left: "15%",
      top: "40%",
    },
  },
  {
    id: "cat",
    name: "고양이",
    image: "/static/characters_transparent/cat_transparent.png",
    position: {
      left: "30%",
      top: "48%",
    },
    size: 220
  },
  {
    id: "bear",
    name: "곰",
    image: "/static/characters_transparent/bear_transparent.png",
    position: {
      left: "58%",
      top: "30%",
    },
  },
  {
    id: "hamster",
    name: "햄스터",
    image: "/static/characters_transparent/hamster_transparent.png",
    position: {
      left: "45%",
      top: "55%",
    },
    size: 200
  },
  {
    id: "raccoon",
    name: "너구리",
    image: "/static/characters_transparent/raccoon_transparent.png",
    position: {
      left: "40%",
      top: "30%",
    },
    size: 230,
    imageStyle: { marginBottom: '-40px' },
  },
  {
    id: "sheep",
    name: "강아지",
    image: "/static/characters_transparent/sheep_transparent.png",
    position: {
      left: "65%",
      top: "58%",
    },
    size: 200
  },
  
];