# Face Crop API

This API provides a face detection and cropping service using the Pigo face detection library.

## Features

- Detects faces in an image from a given URL.
- Crops the detected face region with configurable zoom and size parameters.
- Returns the cropped face image as a JPEG.
- If no face is detected, returns a default face image.

## API Endpoint

### GET /crop

#### Query Parameters

- `src` (string, required): Base64-encoded URL of the source image to process.
- `zoom` (float, optional): Zoom factor for the face crop. Default is 1.0.
- `size` (float, optional): Resize factor for the cropped face image. Default is 1.0.

#### Response

- Returns a JPEG image of the cropped face.
- If no face is detected, returns a default face JPEG image.

## Usage

Send a GET request to `/crop` with the required query parameters. Example:

```
GET /crop?src=<base64-encoded-image-url>&zoom=1.2&size=1.0
```

## Implementation Details

- Uses the Pigo library for face detection.
- Converts the input image to grayscale for detection.
- Detects faces and clusters detections.
- Crops the first detected face with zoom and size adjustments.
- Resizes the cropped image using Lanczos resampling.
- Handles errors gracefully and returns appropriate HTTP status codes.

## Running the Server

Run the Go application, which starts an HTTP server on port 8081:

```
go run main.go
```

The server listens on `http://localhost:8081`.

## Dependencies

- [Pigo](https://github.com/esimov/pigo) for face detection.
- [nfnt/resize](https://github.com/nfnt/resize) for image resizing.

## Notes

- The face detection cascade file (`facefinder`) is a cascade classifier used by the Pigo library to detect faces.
- This cascade file is embedded into the application binary using Go's `embed` package, allowing the app to be distributed as a single executable without external dependencies.
- CORS is enabled for all origins.

## Build Instructions

To build the application, ensure you have Go installed and run:

```
go build -o face-cropr main.go
```

This will produce a single binary executable `face-cropr` that includes the embedded cascade file.

---

> This project is actually a result of Vibe Coding with Gemini Pro.
