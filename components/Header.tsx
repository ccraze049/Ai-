
import React from 'react';

export const Header: React.FC = () => {
  return (
    <header className="text-center">
      <h1 className="text-4xl sm:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-500">
        hk Ai video Generator
      </h1>
      <p className="mt-4 text-lg text-gray-300">
        Bring your images to life with the power of AI animation.
      </p>
    </header>
  );
};
