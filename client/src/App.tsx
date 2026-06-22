import { useState, useRef, useEffect } from "react";
import { GameSidebar } from "./components/GameSideBar";
import { OutfitPreviewModal } from "./components/OutfitPreviewModal";
import "./app.css";
import "./components/sidebar.css";
import { characters } from "./types/CharacterType";
import { Character } from "./components/Character";
import type { JobStatus } from "./types/JobTypes";

export default function OutfitGame() {
  const [selectedCharacterId, setSelectedCharacterId] = useState("bear");

  const [description, setDescription] = useState(
    "곰 캐릭터가 여름에 수영을 한다. 파란색 수영복을 추천해줘.",
  );

  const bgr = "/static/backgrounds/background.png"

  const [recommendationCount, setRecommendationCount] = useState(1);

  const [isGenerating, setIsGenerating] = useState(false);

  const [job, setJob] = useState<JobStatus | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [previewCharacterId, setPreviewCharacterId] = useState<string | null>(null);
  const [characterImages, setCharacterImages] = useState<Record<string, string>>({});
  const [isApplying, setIsApplying] = useState(false);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handleGenerate = async () => {
    if (!description.trim()) return;
    if (pollRef.current) clearInterval(pollRef.current);

    try {
      setIsGenerating(true);
      setJob({ status: "pending", progress: [], results: null, error: null });

      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_text: description,
          character_id: selectedCharacterId,
          top_n: recommendationCount,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        alert(err.detail ?? "오류가 발생했어요");
        setIsGenerating(false);
        setJob(null);
        return;
      }

      const { job_id } = await res.json();

      const poll = async () => {
        try {
          const statusRes = await fetch(`/api/jobs/${job_id}`);
          const data: JobStatus = await statusRes.json();
          setJob(data);
          if (data.status === "completed" || data.status === "failed") {
            if (pollRef.current) clearInterval(pollRef.current);
            setIsGenerating(false);
          }
        } catch (e) {
          console.error(e);
        }
      };

      poll();
      pollRef.current = setInterval(poll, 2000);
    } catch (e) {
      alert(`서버에 연결할 수 없어요: ${e}`);
      setIsGenerating(false);
      setJob(null);
    }
  };

  return (
    <div className="outfit-game">
      <section className="outfit-game__world">
        <img
          className="outfit-game__background"
          src={bgr}
          alt=""
        />

        {characters.map((character) => (
          <Character
            key={character.id}
            name={character.name}
            image={characterImages[character.id] ?? character.image}
            selected={selectedCharacterId === character.id}
            onClick={() => setSelectedCharacterId(character.id)}
            size={character.size}
            imageStyle={character.imageStyle}
            style={{
              left: character.position.left,
              top: character.position.top,
            }}
          />
        ))}
      </section>

      <GameSidebar
        characters={characters}
        selectedCharacterId={selectedCharacterId}
        description={description}
        recommendationCount={recommendationCount}
        isGenerating={isGenerating}
        job={job}
        onCharacterChange={setSelectedCharacterId}
        onDescriptionChange={setDescription}
        onRecommendationCountChange={setRecommendationCount}
        onGenerate={handleGenerate}
        onResultClick={(imageUrl) => {
          setPreviewImage(imageUrl);
          setPreviewCharacterId(selectedCharacterId);
        }}
      />

      {previewImage && (
        <OutfitPreviewModal
          imageUrl={previewImage}
          onClose={() => { setPreviewImage(null); setPreviewCharacterId(null); }}
          isApplying={isApplying}
          onApply={async () => {
            setIsApplying(true);
            try {
              const res = await fetch("/api/make-transparent", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_url: previewImage }),
              });
              if (!res.ok) throw new Error("배경 제거 실패");
              const { transparent_url } = await res.json();
              const targetId = previewCharacterId ?? selectedCharacterId;
              setCharacterImages((prev) => ({
                ...prev,
                [targetId]: transparent_url,
              }));
            } catch {
              const targetId = previewCharacterId ?? selectedCharacterId;
              setCharacterImages((prev) => ({
                ...prev,
                [targetId]: previewImage,
              }));
            } finally {
              setIsApplying(false);
              setPreviewImage(null);
              setPreviewCharacterId(null);
            }
          }}
        />
      )}
    </div>
  );
}