import React from 'react';

interface ModeIndicatorProps {
  appMode: "coding" | "non-coding";
}

const ModeIndicator: React.FC<ModeIndicatorProps> = ({ appMode }) => {
  return (
    <div className="flex items-center gap-2 px-2 py-1 bg-black/30 border border-white/10 rounded-lg">
      <div className="flex items-center gap-1">
        <div 
          className={`w-2 h-2 rounded-full ${
            appMode === "coding" ? "bg-green-400" : "bg-blue-400"
          }`}
        />
        <span className="text-xs text-white/80 font-medium">
          {appMode === "coding" ? "Coding" : "Non-Coding"}
        </span>
      </div>
      <div className="text-xs text-white/60">
        Ctrl+N to switch
      </div>
    </div>
  );
};

export default ModeIndicator;
