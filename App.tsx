
import React, { useState, useCallback } from 'react';
import { ImageUploader } from './components/ImageUploader';
import { InstructionInput } from './components/InstructionInput';
import { ActionButton } from './components/ActionButton';
import { VideoPlayer } from './components/VideoPlayer';
import { Loader } from './components/Loader';
import { generateVideo } from './services/geminiService';
import { Header } from './components/Header';
import { ErrorDisplay } from './components/ErrorDisplay';

const App: React.FC = () => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImageChange = useCallback((file: File | null) => {
    setImageFile(file);
    setVideoUrl(null);
    setError(null);
  }, []);

  const handlePromptChange = useCallback((value: string) => {
    setPrompt(value);
  }, []);

  const handleAnimate = async () => {
    if (!imageFile || !prompt) {
      setError('Please upload an image and provide animation instructions.');
      return;
    }
    setError(null);
    setVideoUrl(null);
    setIsLoading(true);

    try {
      const url = await generateVideo(prompt, imageFile);
      setVideoUrl(url);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred.';
      setError(`Video generation failed: ${errorMessage}`);
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = () => {
    if (!videoUrl) return;
    const a = document.createElement('a');
    a.href = videoUrl;
    a.download = 'hk-ai-generated-video.mp4';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(videoUrl);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 font-sans p-4 sm:p-6 lg:p-8">
      <div className="max-w-3xl mx-auto">
        <Header />
        <main className="mt-8 space-y-12">
          <div className="bg-gray-800 p-8 rounded-2xl shadow-xl border border-gray-700 transition-all duration-300 hover:border-indigo-500">
            <h2 className="text-xl sm:text-2xl font-bold text-indigo-400 mb-1">Step 1: Upload Image</h2>
            <p className="text-gray-400 mb-6">Upload Your Starting Image (JPG or PNG)</p>
            <ImageUploader onImageChange={handleImageChange} />
          </div>
          
          <div className="bg-gray-800 p-8 rounded-2xl shadow-xl border border-gray-700 transition-all duration-300 hover:border-indigo-500">
            <h2 className="text-xl sm:text-2xl font-bold text-indigo-400 mb-1">Step 2: Describe Animation</h2>
            <p className="text-gray-400 mb-6">Provide detailed instructions for the AI.</p>
            <InstructionInput value={prompt} onChange={handlePromptChange} />
          </div>

          <div className="flex justify-center">
            <ActionButton
              onClick={handleAnimate}
              disabled={!imageFile || !prompt || isLoading}
            >
              Animate Image
            </ActionButton>
          </div>
          
          {error && <ErrorDisplay message={error} />}

          {isLoading && <Loader />}

          {videoUrl && !isLoading && (
            <div className="bg-gray-800 p-8 rounded-2xl shadow-xl border border-gray-700">
              <h2 className="text-xl sm:text-2xl font-bold text-indigo-400 mb-6 text-center">Your Animated Video</h2>
              <VideoPlayer src={videoUrl} />
              <div className="mt-6 flex justify-center">
                <ActionButton onClick={handleDownload}>
                  Download Video
                </ActionButton>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default App;

