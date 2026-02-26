# Adversarial Attack Assessment — FGSM

---

## 1. How to Run Locally

### Backend Setup

```bash
# Navigate to the directory
cd backend

# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python3 -m uvicorn app_fgsm:app --reload --port 8000
```

### Frontend Setup

```bash
# Navigate to the directory
cd frontend

# Install packages
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## 2. Deployed URLs

| Service | URL |
|---|---|
| Frontend (Next.js) | https://main.d3ezrkzxofwhvz.amplifyapp.com |
| Backend API (FastAPI) | https://chicly-unhesitative-jamila.ngrok-free.dev/docs#/ |

> **Note:** The backend uses an ngrok secure tunnel to bridge the HTTPS Amplify frontend with the HTTP EC2 instance, resolving browser Mixed Content restrictions.

---

## 3. Explanation of FGSM

FGSM is an adversarial attack which is designed to trick neural networks by making small changes to input data. In standard training we adjust the model weights to minimize loss, but in FGSM, we freeze the weights and utilize gradient to change each individual pixel to make the model's error as large as possible. This is done by the formula:

$$x_{new} = x_{old} + \epsilon \cdot \text{SIGN}(\nabla_x J(\theta, x, y))$$

Here it generates the new image tensor by adding the old image matrix to the epsilon (which controls the magnitude of the perturbation), which is multiplied by the SIGN function, which further uses the gradient function's returned matrix and normalizes it into +1 or -1 based on the direction.

By moving the image pixels in the direction that increases the error, the attack forces the model to misclassify the input. This method is highly efficient because it only requires a single gradient calculation, making it a "fast" way to expose vulnerabilities in AI models.

---

## 4. Observations

In my evaluation script (`evaluate.py`), I tested the robustness of a model trained on the MNIST dataset, which consists of 60,000 training images and 10,000 testing images of handwritten digits.

As shown in the execution screenshots, the test accuracy dropped from a baseline of nearly **99%** to approximately **65.24%** when subjected to an FGSM attack with an epsilon of `0.1`.

The accuracy further decreases as we increase the epsilon value, but a bigger epsilon value makes the changes visible to the human eye as well. At an epsilon value of `0.3` I noticed that most of my attacks were successful. At `0.1` the model sometimes predicts the correct answer.

---

## 5. Deployment Choice

**Chosen: Option B — FastAPI on EC2 t2.micro**

1. I had previous experience with EC2.

2. PyTorch has heavy dependencies and Lambda has a strict 250MB deployment package limit (unzipped). Getting PyTorch + FastAPI + image processing libraries under that limit requires extra work like custom Lambda layers or Docker container images. EC2 sidesteps this entirely — you just `pip install` everything and it works.

3. On EC2 you load the pretrained model once at startup and it stays in memory. On Lambda, depending on traffic, the model may reload on every cold start.
