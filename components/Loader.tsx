
import React, { useState, useEffect } from 'react';

const loadingMessages = [
  "Initializing AI animation engine...",
  "Analyzing your image and instructions...",
  "Storyboarding the animation sequence...",
  "Rendering video frames... this can take a few minutes.",
  "Compositing audio and visuals...",
  "Finalizing your masterpiece..."
];

export const Loader: React.FC = () => {
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setMessageIndex((prevIndex) => (prevIndex + 1) % loadingMessages.length);
    }, 4000); // Change message every 4 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col items-center justify-center space-y-4 p-8 bg-gray-800 rounded-2xl border border-gray-700">
      <div className="w-16 h-16 border-4 border-t-indigo-500 border-gray-600 rounded-full animate-spin"></div>
      <p className="text-lg text-gray-300 text-center transition-opacity duration-500">
        {loadingMessages[messageIndex]}
      </p>
    </div>
  );
};
