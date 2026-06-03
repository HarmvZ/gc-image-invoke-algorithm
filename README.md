# Grand Challenge Image Invoke Algorithm

A test algorithm for [Grand Challenge](https://grand-challenge.org) that uses the **invoke API method**. The container stays running and processes invocations via HTTP rather than executing once per job.

## What it does

Takes a generic medical image (SimpleITK `.mha`) as input and outputs a binary overlay of the same spatial dimensions containing a 3D sphere centered in the image. The sphere radius is approximately 1/4 of the smallest image dimension.

- Input: `generic-medical-image` (`.mha`)
- Output: `generic-overlay` (`.mha`, uint8, 0 background / 255 sphere)

## Invoke mode

This algorithm uses the Grand Challenge invoke method (`LABEL org.grand-challenge.api-method="invoke"`). Instead of running once and exiting, it starts a FastAPI/uvicorn HTTP server on port 4743 that exposes:

- `GET /health` → 200 when ready
- `POST /invoke` → reads from `/input`, writes to `/output`, returns 201

The [sagemaker-shim](https://github.com/DIAGNijmegen/rse-sagemaker-shim) manages the container lifecycle and calls these endpoints.

## Project structure

```
├── app.py              # FastAPI server (invoke mode entrypoint)
├── inference.py        # Inference logic (also runnable standalone for exec mode)
├── Dockerfile
├── requirements.txt
├── do_build.sh         # Build the Docker image
├── do_test_run.sh      # Local test (runs inference.py in exec mode)
├── do_save.sh          # Export container image for upload
├── model/              # Model weights directory (mounted at /opt/ml/model)
└── test/input/         # Test input data
```

## Local testing

```bash
./do_test_run.sh
```

This builds the container and runs `inference.py` directly (exec mode) for quick local validation. Output goes to `test/output/`.

## Deploying to Grand Challenge

```bash
./do_save.sh
```

Upload the resulting `.tar.gz` to your algorithm's Containers page on Grand Challenge. The platform will use the invoke method based on the Docker label.

## Based on

- [Grand Challenge algorithm documentation](https://grand-challenge.org/documentation/create-your-own-algorithm/)
- [test-invoke-algorithm](https://github.com/koopmant/test-invoke-algorithm) reference implementation
