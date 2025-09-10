// FIX: Import `GenerateVideosOperation` to correctly type the video generation operation object.
import { GoogleGenAI, GenerateVideosOperation } from "@google/genai";

const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      const result = reader.result as string;
      // Remove the data URL prefix
      const base64 = result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = (error) => reject(error);
  });
};

// FIX: Use the correct `GenerateVideosOperation` type for the operation parameter and return type to resolve type mismatch errors during polling.
const pollOperation = async (
    ai: GoogleGenAI,
    operation: GenerateVideosOperation,
    updateLoadingMessage: (message: string) => void
  ): Promise<GenerateVideosOperation> => {
  let currentOperation = operation;
  const messages = [
    "Analyzing your image and instructions...",
    "Storyboarding the animation sequence...",
    "Rendering video frames... this may take a few minutes.",
    "Compositing audio and visuals...",
    "Applying final touches..."
  ];
  let messageIndex = 0;

  while (!currentOperation.done) {
    updateLoadingMessage(messages[messageIndex % messages.length]);
    messageIndex++;
    await new Promise(resolve => setTimeout(resolve, 10000)); // Poll every 10 seconds
    try {
      currentOperation = await ai.operations.getVideosOperation({ operation: currentOperation });
    } catch (error) {
       console.error("Error polling for video operation status:", error);
       throw new Error("Failed to get video generation status.");
    }
  }
  return currentOperation;
};

export const generateVideo = async (prompt: string, imageFile: File): Promise<string> => {
  if (!process.env.API_KEY) {
    throw new Error("API_KEY environment variable not set.");
  }

  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const imageBase64 = await fileToBase64(imageFile);

  let initialOperation = await ai.models.generateVideos({
    model: 'veo-2.0-generate-001',
    prompt: prompt,
    image: {
      imageBytes: imageBase64,
      mimeType: imageFile.type,
    },
    config: {
      numberOfVideos: 1,
    }
  });

  // FIX: Removed the unnecessary and incorrect generic type argument from the `pollOperation` call.
  const finalOperation = await pollOperation(ai, initialOperation, () => {}); // A no-op message updater as Loader handles it.

  const downloadLink = finalOperation.response?.generatedVideos?.[0]?.video?.uri;

  if (!downloadLink) {
    throw new Error("Video generation succeeded, but no download link was provided.");
  }
  
  const response = await fetch(`${downloadLink}&key=${process.env.API_KEY}`);
  if (!response.ok) {
    throw new Error(`Failed to download video file: ${response.statusText}`);
  }
  
  const videoBlob = await response.blob();
  const blobUrl = URL.createObjectURL(videoBlob);
  return blobUrl;
};
