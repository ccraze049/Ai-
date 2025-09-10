
import React from 'react';

interface InstructionInputProps {
  value: string;
  onChange: (value: string) => void;
}

const placeholderText = `Describe how you want to animate the image.
For example: 'Make the waterfall move and add a subtle mist effect,'
or 'Slowly zoom in on the person's face and make them blink.'
If you want dialogue, write it here — specify the exact text and language.
Example (Hindi):
'Character should say in Hindi: मेरा नाम राम है, इस वीडियो में आपका स्वागत है।'`;

export const InstructionInput: React.FC<InstructionInputProps> = ({ value, onChange }) => {
  return (
    <textarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholderText}
      rows={8}
      className="w-full p-4 bg-gray-900 border border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition duration-200 text-gray-200 placeholder-gray-500 text-base resize-y"
    />
  );
};
