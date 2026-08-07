export const EMOTION_PRESETS = [
  'neutral',
  'happy',
  'sad',
  'angry',
  'excited',
  'calm',
  'fearful',
  'tender',
] as const;

export const TONE_PRESETS = [
  'natural',
  'warm',
  'soft',
  'confident',
  'serious',
  'dramatic',
  'energetic',
  'storytelling',
] as const;

export const INTENSITY_PRESETS = ['subtle', 'medium', 'strong'] as const;

export type EmotionPreset = (typeof EMOTION_PRESETS)[number];
export type TonePreset = (typeof TONE_PRESETS)[number];
export type IntensityPreset = (typeof INTENSITY_PRESETS)[number];

const MAX_INSTRUCT_LENGTH = 500;

const EMOTION_INSTRUCTIONS: Record<EmotionPreset, string> = {
  neutral: '',
  happy: 'Speak with a happy, upbeat emotion.',
  sad: 'Speak with a sad, emotionally restrained delivery.',
  angry: 'Speak with an angry, forceful emotion.',
  excited: 'Speak with an excited, enthusiastic emotion.',
  calm: 'Speak with a calm, composed emotion.',
  fearful: 'Speak with a fearful, tense emotion.',
  tender: 'Speak with a tender, affectionate emotion.',
};

const TONE_INSTRUCTIONS: Record<TonePreset, string> = {
  natural: '',
  warm: 'Use a warm and inviting tone.',
  soft: 'Use a soft and gentle tone.',
  confident: 'Use a confident and assured tone.',
  serious: 'Use a serious and grounded tone.',
  dramatic: 'Use a dramatic, cinematic tone.',
  energetic: 'Use an energetic and lively tone.',
  storytelling: 'Use a natural storytelling cadence with expressive phrasing.',
};

const INTENSITY_INSTRUCTIONS: Record<IntensityPreset, string> = {
  subtle: 'Keep the expression subtle and natural.',
  medium: 'Make the expression clearly noticeable while keeping it natural.',
  strong: 'Make the expression strong without distorting pronunciation or becoming unnatural.',
};

interface BuildVoiceStyleInstructionOptions {
  emotion?: EmotionPreset;
  tone?: TonePreset;
  intensity?: IntensityPreset;
  customInstruction?: string;
}

/**
 * Compile engine-agnostic Voicebox style controls into Qwen CustomVoice's
 * existing natural-language `instruct` parameter.
 */
export function buildVoiceStyleInstruction({
  emotion = 'neutral',
  tone = 'natural',
  intensity = 'medium',
  customInstruction = '',
}: BuildVoiceStyleInstructionOptions): string | undefined {
  const automaticParts = [EMOTION_INSTRUCTIONS[emotion], TONE_INSTRUCTIONS[tone]].filter(Boolean);

  if (automaticParts.length > 0) {
    automaticParts.push(INTENSITY_INSTRUCTIONS[intensity]);
  }

  const automaticInstruction = automaticParts.join(' ').trim();
  const custom = customInstruction.trim();

  if (!automaticInstruction && !custom) return undefined;
  if (!automaticInstruction) return custom.slice(0, MAX_INSTRUCT_LENGTH);
  if (!custom) return automaticInstruction.slice(0, MAX_INSTRUCT_LENGTH);

  const separator = ' ';
  const availableCustomLength = Math.max(
    0,
    MAX_INSTRUCT_LENGTH - automaticInstruction.length - separator.length,
  );

  return `${automaticInstruction}${separator}${custom.slice(0, availableCustomLength)}`.trim();
}
