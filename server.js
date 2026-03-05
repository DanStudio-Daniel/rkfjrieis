// server.js
const express = require('express');
const { LlamaCpp, LlamaLogLevel } = require('node-llama-cpp');
const path = require('path');
const fs = require('fs-extra');

const app = express();
const port = process.env.PORT || 10000;

app.use(express.json());

// --- Model Configuration ---
const MODEL_DIR = path.join(process.cwd(), 'models');
const MODEL_NAME = 'tiny-vicuna-1b.q2_k.gguf';
const MODEL_PATH = path.join(MODEL_DIR, MODEL_NAME);

let llm = null;

async function loadModel() {
    if (llm === null) {
        console.log(`[${new Date().toISOString()}] Attempting to load model from: ${MODEL_PATH}`);

        if (!(await fs.pathExists(MODEL_PATH))) {
            const errorMsg = `Model file not found at ${MODEL_PATH}. Ensure it was downloaded during the build process.`;
            console.error(`[${new Date().toISOString()}] Error: ${errorMsg}`);
            throw new Error(errorMsg);
        }

        try {
            llm = new LlamaCpp({
                modelPath: MODEL_PATH,
                // nGpuLayers: 0,
                nCtx: 512,
                nBatch: 512,
                // logLevel: LlamaLogLevel.debug,
            });
            console.log(`[${new Date().toISOString()}] Model loaded successfully!`);
        } catch (error) {
            console.error(`[${new Date().toISOString()}] Error initializing LlamaCpp:`, error);
            throw new Error(`Failed to initialize LLM model: ${error.message}`);
        }
    }
    return llm;
}

// --- API Endpoints ---

app.get('/', (req, res) => {
    res.status(200).send('LLM service is running (Node.js). Send POST requests to /generate for text generation.');
});

app.get('/models', async (req, res) => {
    try {
        const files = await fs.readdir(MODEL_DIR);
        const ggufModels = files.filter(file => file.endsWith('.gguf'));
        if (ggufModels.length === 0) {
            return res.status(404).json({ message: `No .gguf models found in ${MODEL_DIR}` });
        }
        res.json({ models: ggufModels, modelDirectory: MODEL_DIR });
    } catch (error) {
        if (error.code === 'ENOENT') {
            return res.status(404).json({ message: `Model directory not found: ${MODEL_DIR}. Has the build command run correctly?` });
        }
        console.error(`[${new Date().toISOString()}] Error listing models:`, error);
        res.status(500).json({ error: `Failed to list models: ${error.message}` });
    }
});

app.post('/generate', async (req, res) => {
    const { prompt, max_tokens = 100, temperature = 0.7 } = req.body;

    if (!prompt) {
        return res.status(400).json({ error: 'Prompt is required in the request body.' });
    }

    try {
        const llmInstance = await loadModel();
        if (!llmInstance) {
            return res.status(500).json({ error: 'LLM model failed to initialize or is not available.' });
        }

        console.log(`[${new Date().toISOString()}] Request: Prompt='${prompt.substring(0, 80)}${prompt.length > 80 ? '...' : ''}', MaxTokens=${max_tokens}, Temp=${temperature}`);

        const completion = await llmInstance.createCompletion({
            prompt: prompt,
            nPredict: max_tokens,
            temperature: temperature,
            stop: ["\n", "User:", "###"],
        });

        const generatedText = completion.trim();
        console.log(`[${new Date().toISOString()}] Response: Generated='${generatedText.substring(0, 80)}${generatedText.length > 80 ? '...' : ''}'`);
        res.json({ generated_text: generatedText });

    } catch (error) {
        console.error(`[${new Date().toISOString()}] Generation error:`, error);
        res.status(500).json({ error: `Text generation failed: ${error.message}` });
    }
});

// --- Application Entry Point ---
loadModel().then(() => {
    app.listen(port, () => {
        console.log(`[${new Date().toISOString()}] Node.js LLM API listening on port ${port}`);
    });
}).catch(err => {
    console.error(`[${new Date().toISOString()}] Critical startup error:`, err);
    process.exit(1);
});
