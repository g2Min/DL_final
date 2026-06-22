import type { ChangeEvent } from "react";
import type { JobStatus } from "../types/JobTypes";
import leaf from "../assets/leaf.png";
import clothes from "../assets/clothes.png";
import kk from "../assets/kk_face.png";
import raccoon from "../assets/raccoon_face.png";

export type CharacterOption = {
  id: string;
  name: string;
  image: string;
};

type GameSidebarProps = {
  characters: CharacterOption[];
  selectedCharacterId: string;
  description: string;
  recommendationCount: number;
  isGenerating?: boolean;
  job?: JobStatus | null;

  onCharacterChange: (characterId: string) => void;
  onDescriptionChange: (value: string) => void;
  onRecommendationCountChange: (count: number) => void;
  onGenerate: () => void;
  onResultClick?: (imageUrl: string) => void;
};

export function GameSidebar({
  characters,
  selectedCharacterId,
  description,
  recommendationCount,
  isGenerating = false,
  job,
  onCharacterChange,
  onDescriptionChange,
  onRecommendationCountChange,
  onGenerate,
  onResultClick,
}: GameSidebarProps) {
  const selectedCharacter =
    characters.find(
      (character) => character.id === selectedCharacterId,
    ) ?? characters[0];

  const handleDescriptionChange = (
    event: ChangeEvent<HTMLTextAreaElement>,
  ) => {
    onDescriptionChange(event.target.value);
  };

  return (
    <aside className="game-sidebar">
      <div className="game-sidebar__paper">
        <header className="game-sidebar__header">
          <span className="game-sidebar__leaf">
            <img src={leaf} style={{ width: "1.5em", height: "2em", display: "block" }}/>
          </span>

          <div>
            <h1>Animal Outfit</h1>
            <h2>Generator</h2>
            <p>동물 친구를 위한 완벽한 의상을 추천해요!</p>
          </div>

          <span className="game-sidebar__star">✦</span>
        </header>

        <section className="game-panel">
          <h3 className="game-panel__title">캐릭터 선택</h3>

          <div className="selected-character-card">
            <button
              type="button"
              className="selected-character-card__arrow"
              aria-label="이전 캐릭터"
              onClick={() => {
                const currentIndex = characters.findIndex(
                  (character) =>
                    character.id === selectedCharacterId,
                );

                const previousIndex =
                  currentIndex <= 0
                    ? characters.length - 1
                    : currentIndex - 1;

                onCharacterChange(
                  characters[previousIndex].id,
                );
              }}
            >
              ‹
            </button>

            <div className="selected-character-card__portrait">
              <img
                src={selectedCharacter.image}
                alt={selectedCharacter.name}
                draggable={false}
              />
            </div>

            <div className="selected-character-card__content">
              <strong>{selectedCharacter.name}</strong>

              <span style={{ display: "inline-flex", alignItems: "center", gap: "0.25em", paddingRight: 20 }}>
                <span aria-hidden="true">
                  <img src={leaf} style={{ width: "1.5em", height: "2em" }}/>
                </span>
                선택된 <br/> 캐릭터
              </span>
            </div>

            <button
              type="button"
              className="selected-character-card__arrow"
              aria-label="다음 캐릭터"
              onClick={() => {
                const currentIndex = characters.findIndex(
                  (character) =>
                    character.id === selectedCharacterId,
                );

                const nextIndex =
                  currentIndex >= characters.length - 1
                    ? 0
                    : currentIndex + 1;

                onCharacterChange(
                  characters[nextIndex].id,
                );
              }}
            >
              ›
            </button>
          </div>
        </section>

        <section className="game-panel">
          <label
            className="game-panel__title"
            htmlFor="outfit-description"
          >
            상황 설명
          </label>

          <div className="game-textarea-wrapper">
            <textarea
              id="outfit-description"
              value={description}
              onChange={handleDescriptionChange}
              placeholder="예: 곰 캐릭터가 여름에 수영을 해. 파란색 수영복을 추천해줘."
              maxLength={300}
            />

            <div className="game-textarea-decoration">
              <img src={raccoon} style={{ width: "1.5em", height: "1.5em", display: "flex"}} />
              <img src={kk} style={{ width: "1.5em", height: "1.5em", display: "flex"}}/>
            </div>
          </div>
        </section>

        <section className="game-panel">
          <h3 className="game-panel__title">
            추천 조합 개수
          </h3>

          <div className="count-selector">
            {[1, 2, 3, 4, 5].map((count) => (
              <button
                key={count}
                type="button"
                className={
                  recommendationCount === count
                    ? "count-selector__button is-active"
                    : "count-selector__button"
                }
                onClick={() =>
                  onRecommendationCountChange(count)
                }
              >
                {count}
              </button>
            ))}
          </div>
        </section>

        <button
          type="button"
          className="generate-button"
          onClick={onGenerate}
          disabled={isGenerating}
        >
          <span className="generate-button__sparkle">
            ✦
          </span>

          <span className="generate-button__icon">
            <img src={clothes} style={{ width: "1.5em", height: "2em", display: "flex"}}/>
          </span>

          <span>
            {isGenerating
              ? "의상 생성 중..."
              : "의상 생성하기"}
          </span>

          <span className="generate-button__sparkle">
            ✦
          </span>
        </button>

        <section className="result-panel">
          <h3 className="game-panel__title">생성 결과</h3>

          {(!job || job.status === "pending") && (
            <div className="result-card">
              <div className="result-card__avatar">
                <img
                  src={selectedCharacter.image}
                  alt=""
                  draggable={false}
                />
              </div>

              <div className="result-card__content">
                <strong>
                  {job
                    ? "생성 준비 중..."
                    : "아직 생성된 의상이 없어요"}
                </strong>

                <p>
                  {job
                    ? "잠시만 기다려 주세요."
                    : "상황을 입력한 뒤 의상 생성 버튼을 눌러주세요."}
                </p>
              </div>
            </div>
          )}

          {job?.status === "running" && (
            <div className="result-card is-running">
              <div className="result-card__content">
                <strong>의상 생성 중...</strong>

                <p>
                  {job.progress[job.progress.length - 1] ??
                    "처리 중..."}
                </p>
              </div>
            </div>
          )}

          {job?.status === "completed" &&
            job.results &&
            job.results.length > 0 && (
              <div className="result-images">
                {job.results.map((r) => (
                  <div
                    key={r.combo_idx}
                    className="result-image-item"
                    role="button"
                    tabIndex={0}
                    style={{ cursor: 'pointer' }}
                    onClick={() => onResultClick?.(r.image_url)}
                    onKeyDown={(e) => e.key === 'Enter' && onResultClick?.(r.image_url)}
                  >
                    <img
                      src={r.image_url}
                      alt={`조합 ${r.combo_idx + 1}`}
                    />

                    <span>조합 {r.combo_idx + 1}</span>
                  </div>
                ))}
              </div>
            )}

          {job?.status === "completed" &&
            (!job.results || job.results.length === 0) && (
              <div className="result-card is-running">
                <div className="result-card__content">
                  <strong>추천 완료</strong>
                  <p>생성된 이미지가 없습니다.</p>
                </div>
              </div>
            )}

          {job?.status === "failed" && (
            <div className="result-card is-failed">
              <div className="result-card__content">
                <strong>생성 실패</strong>
                <p>
                  {job.error ?? "알 수 없는 오류가 발생했어요."}
                </p>
              </div>
            </div>
          )}
        </section>

        <footer className="game-sidebar__footer">
          <span>✦</span>
          <span>의상 추천 시스템</span>
          <span>✦</span>
        </footer>
      </div>
    </aside>
  );
}