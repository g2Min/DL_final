import './outfit-preview-modal.css';

type Props = {
  imageUrl: string;
  onClose: () => void;
  onApply: () => void;
  isApplying?: boolean;
};

export function OutfitPreviewModal({ imageUrl, onClose, onApply, isApplying = false }: Props) {
  return (
    <div className="outfit-modal-overlay" onClick={onClose}>
      <div className="outfit-modal" onClick={(e) => e.stopPropagation()}>
        <button className="outfit-modal__close" onClick={onClose} aria-label="닫기">
          ✕
        </button>

        <img className="outfit-modal__image" src={imageUrl} alt="생성된 의상" />

        <button
          className="outfit-modal__apply"
          onClick={onApply}
          disabled={isApplying}
        >
          {isApplying ? '⏳ 적용 중...' : '👕 옷 갈아입기'}
        </button>
      </div>
    </div>
  );
}
