export interface UiElements {
  loading: HTMLElement;
  loadingMessage: HTMLElement;
  toast: HTMLElement;
  splitRange: HTMLInputElement;
  splitLine: HTMLElement;
  sseRange: HTMLInputElement;
  sseValue: HTMLOutputElement;
  fps: HTMLElement;
  measureHint: HTMLElement;
  terraResult: HTMLOutputElement;
  a1Result: HTMLOutputElement;
}

export function element<T extends Element>(selector: string): T {
  const match = document.querySelector<T>(selector);
  if (!match) throw new Error(`Required viewer element is missing: ${selector}`);
  return match;
}

export function createUi(): UiElements {
  return {
    loading: element("#loadingScreen"),
    loadingMessage: element("#loadingMessage"),
    toast: element("#toast"),
    splitRange: element("#splitRange"),
    splitLine: element(".split-line"),
    sseRange: element("#sseRange"),
    sseValue: element("#sseValue"),
    fps: element("#globalFps"),
    measureHint: element("#measureHint"),
    terraResult: element("#terraResult"),
    a1Result: element("#a1Result"),
  };
}
