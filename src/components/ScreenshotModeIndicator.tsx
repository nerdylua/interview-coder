import React, { useState, useEffect } from 'react';

const ScreenshotModeIndicator: React.FC = () => {
  const [mode, setMode] = useState<string>('full');

  useEffect(() => {
    // Get initial mode
    window.electronAPI.getScreenshotMode().then(setMode);

    // Listen for mode changes
    const unsubscribe = window.electronAPI.onScreenshotModeChanged((newMode: string) => {
      setMode(newMode);
    });

    return unsubscribe;
  }, []);

  const getModeDisplay = (mode: string) => {
    switch (mode) {
      case 'full':
        return { text: 'Full', icon: '⬜', shortcut: 'Ctrl+1' };
      case 'left':
        return { text: 'Left', icon: '◐', shortcut: 'Ctrl+2' };
      case 'right':
        return { text: 'Right', icon: '◑', shortcut: 'Ctrl+3' };
      default:
        return { text: 'Full', icon: '⬜', shortcut: 'Ctrl+1' };
    }
  };

  const { text, icon, shortcut } = getModeDisplay(mode);

  return (
    <div
      className="flex items-center gap-1 px-1.5 py-0.5 bg-white/5 rounded text-[10px] text-white/60 border border-white/10"
      title={`Screenshot Mode: ${text} (${shortcut})`}
    >
      <span className="text-[11px]">{icon}</span>
      <span className="font-mono">{text}</span>
    </div>
  );
};

export default ScreenshotModeIndicator;