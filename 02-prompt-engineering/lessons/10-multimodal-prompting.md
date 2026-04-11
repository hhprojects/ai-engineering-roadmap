# Lesson 10 — Multimodal Prompting

> **The single sentence version:** Modern LLMs can see images (and some can hear, and watch video), and the techniques for prompting them well are mostly the same as for text — with a handful of multimodal-specific tricks like grounding, cropping, and OCR-first workflows.

Every frontier model in 2026 can take images as input. Some handle audio. A few handle video natively. This chapter covers what's possible today, what prompting techniques apply to multimodal inputs, and where the pitfalls are.

This is the shortest lesson in the module because most of the techniques you already know (clear instructions, examples, structured outputs, CoT) transfer straight to multimodal tasks. What's new is the *data format*, the *specific things to ask about*, and a few tricks that only matter when images are involved.

---

## What "multimodal" actually means

When we say a model is multimodal, we mean it can take more than one kind of input. The usual breakdown:

- **Text + image input → text output** — the most common. Claude, GPT, Gemini, and most recent open-weight models support this.
- **Text + audio input → text output** — speech recognition, transcription, audio analysis. Whisper is the canonical example; Gemini and newer GPT models handle audio natively.
- **Text + video input → text output** — video understanding. Gemini 3 is the only frontier model with first-class native video support as of early 2026.
- **Text input → image output** — text-to-image generation (DALL-E, Imagen, Flux). Different model family, different techniques, not covered in this module.
- **Text + image input → image output** — image editing. A growing category; most mature tools are outside the chat-LLM ecosystem.
- **Text + audio input → audio output** — voice-mode conversations. OpenAI Realtime API, Gemini Live.

For this chapter we focus on the dominant case: **text + image input → text output**. Everything else is either outside the scope (image generation) or a niche for which the same principles apply.

---

## How images get into the prompt

All three major providers support sending images as part of the `user` message content. The specific API shape differs slightly.

### Anthropic (Claude)

```python
import anthropic
import base64

client = anthropic.Anthropic()

with open("receipt.jpg", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Extract the total amount, currency, and line items from this receipt.",
                },
            ],
        }
    ],
)
```

Notice that the content is a *list* of content blocks, not a string. Each block has a type — `image` or `text`. You can have multiple images in one message, and interleave them with text ("Here's the first photo: [image]. Here's the second: [image]. Compare them.").

### OpenAI (GPT)

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the receipt data."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,{base64_image}",
                        "detail": "high",   # or "low" or "auto"
                    },
                },
            ],
        }
    ],
)
```

The `detail` parameter controls the number of tokens the image costs. `"low"` uses ~85 tokens regardless of image size and gives a coarse view; `"high"` uses ~170 tokens for the overview plus additional tokens for 512×512 tiles of the image. Use `"low"` for logos or simple images, `"high"` for anything with dense text or fine detail.

### Google (Gemini)

```python
import google.generativeai as genai

model = genai.GenerativeModel("gemini-3-flash")

response = model.generate_content([
    "What's in this image?",
    {"mime_type": "image/jpeg", "data": image_bytes},
])
```

Gemini's API is structurally similar but uses its own `Parts` convention. You can also pass a file URL or an uploaded file reference, which is how long video input works.

### Sending images from URLs

All three providers also let you pass an image URL instead of base64 bytes, which is much more efficient if the image is already hosted somewhere. Use URLs when you can:

```python
# OpenAI example
{"type": "image_url", "image_url": {"url": "https://example.com/receipt.jpg"}}
```

Caveats: the provider needs to be able to access the URL (no authentication), the URL should be stable (no signed URLs that expire in 5 seconds), and some providers cache URL fetches, so a URL that returns different content each time will confuse you.

---

## What models can see

Modern vision models are remarkably good at a lot of things, and surprisingly bad at a few others. A rough map of the landscape:

**What they're good at:**
- **Object recognition** — "what's in this photo?" works reliably on common objects
- **Scene description** — describing environments, locations, moods
- **OCR (text in images)** — reading printed text is nearly flawless on well-lit photos
- **Diagrams and charts** — reading axis labels, extracting values, interpreting trends
- **Document analysis** — receipts, forms, invoices, screenshots
- **UI analysis** — identifying buttons, forms, errors on screenshots
- **Counting small numbers of things** — "how many people are in this photo?" works up to ~10
- **Spatial relationships at a coarse level** — "is the dog to the left or right of the car?"

**What they struggle with:**
- **Precise counting** — "exactly how many birds are on that wire?" is unreliable for >10 items
- **Fine spatial measurements** — "how far is X from Y in pixels?" is usually wrong
- **Dense tables** — a 50-row spreadsheet screenshot often loses rows or mangles values
- **Rotated / skewed text** — performs worse as the angle increases
- **Subtle color distinctions** — "is this exactly #FF5733 or #FF5734?" no
- **Abstract visual reasoning puzzles** — Raven's Progressive Matrices and similar
- **Handwriting** — much worse than printed text, especially cursive or messy
- **Temporal reasoning from a single frame** — "is this person about to fall?" is speculation
- **Counting pixels or exact dimensions** — unreliable

Knowing what they can and can't do shapes how you prompt. If you need to count 30 items, don't ask the model to count them all — ask it to describe them in rows and count the rows yourself, or break the image into quadrants and count each.

---

## Principles for prompting with images

Most of the text-prompting techniques you've learned still apply. The adjustments are small but important.

### 1. Put the image first

Anthropic's docs specifically recommend placing images *before* the text question in the message content. Their own testing showed that putting the image before the question improves accuracy — sometimes by a meaningful margin.

```python
content = [
    {"type": "image", "source": {...}},      # image first
    {"type": "text",  "text": "Question: ..."}
]
```

Intuition: the model processes the message content sequentially. Putting the image first lets all the text tokens attend back to the image's visual features. Putting the text first means the question "doesn't know" about the image until the model processes both.

### 2. Be specific about what to look for

A vague prompt produces a vague answer. "What's in this photo?" is weaker than "Describe the main subject, any visible text, and anything that looks unusual or out of place."

For extraction tasks, be explicit about fields:

```
Look at this receipt image. Extract:
1. Merchant name
2. Date (in YYYY-MM-DD format)
3. Total amount and currency
4. Each line item with its price

Return as JSON matching this schema: {schema}

If a field is not clearly visible or is unreadable, return null for that field.
Do not guess.
```

Much more reliable than "parse this receipt."

### 3. Use chain-of-thought for visual reasoning

When the task requires reasoning about what's in the image — not just identifying it — ask the model to think through its observations before concluding.

```
Look at the screenshot of this dashboard. Describe what you see step by step:
1. What's the overall layout?
2. What numbers are displayed, and what are they labeled?
3. Any visual alerts, warnings, or error states?
4. Based on your observations, is anything anomalous?
```

Forcing the model to describe before it judges produces more grounded answers. Without this, the model often jumps to a conclusion it then can't defend.

### 4. Ground responses in image quotes

For documents with text, ask the model to quote what it sees before answering questions about it. This reduces hallucinations dramatically.

```
Read this contract screenshot. First, quote the specific clauses relevant to
payment terms in <quote> tags. Then, in <answer> tags, summarize what the
payment terms say.
```

The model now has to *point to* the text before it can summarize it. If the text isn't actually in the image, the model is forced to say so rather than hallucinating.

### 5. Provide context about what the image is

The model doesn't know why you're showing it the image. Tell it.

Weak: "What do you see?"

Strong: "This is a screenshot of a software bug report from one of our users. Tell me:
1. What feature is the user trying to use?
2. What error message (if any) is shown?
3. What would be a useful follow-up question to ask the user?"

The context shapes the model's interpretation enormously.

### 6. Give the model a crop tool (Anthropic-specific)

From Anthropic's docs: on image-heavy tasks, **providing a crop tool** that lets the model zoom in on a specific region of the image measurably improves accuracy. The model decides where to look, calls the crop tool, receives the zoomed image back, and can then reason about the cropped region.

```python
tools = [
    {
        "name": "crop_image",
        "description": "Zoom into a specific region of the image to see more detail.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x_min": {"type": "number", "description": "Left edge (0-1 normalized)"},
                "y_min": {"type": "number", "description": "Top edge (0-1 normalized)"},
                "x_max": {"type": "number", "description": "Right edge (0-1 normalized)"},
                "y_max": {"type": "number", "description": "Bottom edge (0-1 normalized)"},
            },
        },
    }
]
```

The implementation crops the original image to the model's requested region and sends it back as a tool result. Anthropic's cookbook has a worked example. Worth the plumbing for any task involving images with small, dense regions of interest (dashboards, complex diagrams, documents with fine print).

---

## Image-specific extraction patterns

Some multimodal workflows are common enough to deserve their own patterns.

### OCR-first for text-heavy documents

If the document is mostly text (receipts, letters, forms), a two-step pipeline often works better than one-shot vision:

1. **Extract the raw text** from the image in the first LLM call. The prompt is simple: "Extract all the visible text from this image in reading order."
2. **Run a text-only extraction pass** on that raw text with a structured output schema.

Why split this? Two reasons:

- **The raw OCR step is cheaper.** You can use a smaller vision model (Haiku, Gemini Flash) for the extraction, then pass the text to a cheaper text-only model for the structured parsing.
- **Text processing is more reliable.** Schema-constrained extraction on text is nearly perfect; schema-constrained extraction on images is occasionally wrong in surprising ways.

The downside: you lose visual context (layout, color, positioning). For pure text extraction, that's fine. For tasks where layout matters (form fields, tables), you might need both.

### Dashboard screenshot analysis

The prompt pattern that works:

```
This is a screenshot of an analytics dashboard. Identify:
1. Every chart or metric visible, with its label and current value
2. Any time ranges or filters currently applied
3. Any anomalies — metrics that are red, show warning icons, or are outside
   expected ranges

For each identified item, quote the exact text from the dashboard. Do not
paraphrase. If a value is unclear, write "unclear" rather than guessing.
```

Key techniques: explicit enumeration of what to look for, "quote exactly, don't paraphrase," escape hatch for unclear values.

### Comparison across multiple images

Send several images in one message and ask the model to compare:

```python
content = [
    {"type": "image", "source": {...}},   # before
    {"type": "image", "source": {...}},   # after
    {"type": "text",  "text": "What changed between the before and after images?"},
]
```

Models handle this well up to ~5 images; performance degrades on more. For batch comparison, consider summarizing each image individually first, then comparing the summaries.

### Error-state detection in UIs

For screenshot triage in bug reports or QA:

```
Look at this screenshot. I'm debugging a user-reported issue where they say
"the form doesn't work." Answer:
1. What form is visible, if any?
2. Are there any visible error messages, red indicators, or validation warnings?
3. Which fields look empty vs filled?
4. What's a likely cause of the user's issue, based only on what you can see?

If you can't tell from the screenshot alone, say so clearly.
```

Grounding the model in "only what you can see" is the key move — otherwise it'll speculate broadly about what might be wrong.

---

## Cost and token considerations

Images cost tokens. A lot of them. Some rules of thumb:

- **OpenAI GPT-4o**: images cost ~85 tokens at `low` detail, ~170 tokens + tiled overhead at `high`. A typical 1024×1024 image at high detail is ~600-1000 tokens.
- **Anthropic Claude**: images are converted to tokens based on size. A 1024×768 image is roughly 1,000-1,500 tokens. Larger images cost proportionally more.
- **Google Gemini**: images are ~258 tokens each regardless of resolution (for normal-sized images), which can be much cheaper for large images.

Practical consequences:

- **Don't send the full-resolution photo when a thumbnail works.** If your use case is "identify the main object in the image," a 512×512 version is often enough and costs 75% less.
- **Use `detail: "low"` on OpenAI unless you need fine detail.** It's a flat ~85 tokens and good enough for many tasks.
- **Crop before sending.** If you know which region you care about, send just that region.
- **Batch intelligently.** Sending 10 separate one-image requests vs one 10-image request: usually the same total cost, but the 10-image request gives the model the option to reason across images.

---

## Common pitfalls

- **Expecting too much of bad photos.** Blurry, dark, or tiny images produce bad results. Garbage in, garbage out. Pre-process (resize, rotate, enhance) before sending.
- **Asking the model to count >10 things precisely.** It's not reliable. Reformulate to counting by category or batch.
- **Forgetting that models can't see outside the image.** "What's around the corner?" is a hallucination invitation.
- **Not providing an escape hatch.** For extraction tasks, always include "return null if unclear" — otherwise you get hallucinated values.
- **Putting the question before the image.** Anthropic's guidance is the opposite. Image first, question second.
- **Treating vision as magic.** Vision models are *good*, not infallible. For anything safety-critical (medical imaging, legal documents), always have human review.
- **Ignoring the detail parameter on OpenAI.** Running everything at `high` when you don't need it burns tokens.
- **Base64-encoding when URLs are available.** If the image is already on the web, pass the URL — cheaper and easier to debug.
- **Assuming the model can extract from handwriting.** Performance on handwriting varies wildly. Test it before relying on it.
- **Trying to render graphics with the model's text output.** A vision model can *describe* a chart; it cannot *draw* one. That's a different model category.

---

## What to remember from this lesson

- Every frontier LLM in 2026 takes images as input. Claude, GPT, and Gemini are the top choices; Gemini is the only one with first-class native video.
- Put images *before* text in the message content for better performance (Anthropic's guidance).
- Vision models are great at OCR, object recognition, scene description, dashboards, and documents. They struggle with precise counting, dense tables, handwriting, and abstract visual reasoning.
- All the text-prompting techniques transfer: clear instructions, examples, CoT, structured outputs, escape hatches.
- For text-heavy documents, OCR-first (extract text, then process text) is often cheaper and more reliable than one-shot vision.
- Ask the model to quote what it sees before drawing conclusions. Reduces hallucinations.
- Providing a crop tool lets the model zoom in on details — measurably improves accuracy on dense images.
- Images cost tokens. A lot of them. Resize, crop, and use low-detail modes when you can.

Next: prompt injection and jailbreaks — the security chapter you need before you put an LLM in production.

---

## References

- Anthropic, *Vision guide for Claude*. https://docs.claude.com/en/docs/build-with-claude/vision
- Anthropic, *Prompt engineering for images* (crop tool cookbook). https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/claude-prompting-best-practices
- OpenAI, *Vision guide*. https://platform.openai.com/docs/guides/vision
- OpenAI, *Image tokens and detail parameter*. https://platform.openai.com/docs/guides/vision#calculating-costs
- Google, *Gemini vision capabilities*. https://ai.google.dev/gemini-api/docs/vision
- Google, *Gemini video understanding*. https://ai.google.dev/gemini-api/docs/video-understanding

---

[← Lesson 9](09-templates-and-caching.md) | [Back to Prompt Engineering](../README.md) | [Next → Lesson 11: Prompt Injection and Guardrails](11-prompt-injection.md)
