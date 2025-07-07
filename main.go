package main

import (
	"bytes"
	_ "embed"
	"encoding/base64"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	_ "image/png" // Register PNG decoder
	"log"
	"net/http"
	"strconv"

	pigo "github.com/esimov/pigo/core"
	"github.com/nfnt/resize"
)

const defaultFaceBase64 = "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"

//go:embed facefinder
var cascadeFile []byte

// FaceDetector holds the Pigo classifier.
type FaceDetector struct {
	classifier *pigo.Pigo
}

// NewFaceDetector creates a new face detector.
func NewFaceDetector(cascade []byte) (*FaceDetector, error) {
	p := pigo.NewPigo()
	classifier, err := p.Unpack(cascade)
	if err != nil {
		return nil, fmt.Errorf("error unpacking cascade file: %w", err)
	}

	return &FaceDetector{classifier: classifier}, nil
}

// DetectFaces finds all faces in an image.
func (fd *FaceDetector) DetectFaces(img image.Image) []pigo.Detection {
	nrgba := image.NewNRGBA(img.Bounds())
	draw.Draw(nrgba, nrgba.Bounds(), img, image.Point{}, draw.Src)

	pixels := pigo.RgbToGrayscale(nrgba)
	cols, rows := nrgba.Bounds().Max.X, nrgba.Bounds().Max.Y

	cParams := pigo.CascadeParams{
		MinSize:     20,
		MaxSize:     2000,
		ShiftFactor: 0.1,
		ScaleFactor: 1.1,
		ImageParams: pigo.ImageParams{
			Pixels: pixels,
			Rows:   rows,
			Cols:   cols,
			Dim:    cols,
		},
	}

	faces := fd.classifier.RunCascade(cParams, 0)
	faces = fd.classifier.ClusterDetections(faces, 0.18)

	log.Printf("Detected %d faces", len(faces))
	return faces
}

// GetFaceRects converts detections to rectangles.
func GetFaceRects(faces []pigo.Detection, qThresh float32) []image.Rectangle {
	var rects []image.Rectangle
	for _, face := range faces {
		if face.Q > qThresh {
			rects = append(rects, image.Rect(
				face.Col-face.Scale/2,
				face.Row-face.Scale/2,
				face.Col+face.Scale/2,
				face.Row+face.Scale/2,
			))
		}
	}
	return rects
}

// cropImage crops an image to a given rectangle.
func cropImage(img image.Image, crop image.Rectangle) (image.Image, error) {
	type subImager interface {
		SubImage(r image.Rectangle) image.Image
	}

	simg, ok := img.(subImager)
	if !ok {
		return nil, fmt.Errorf("image does not support cropping")
	}

	return simg.SubImage(crop), nil
}

func (fd *FaceDetector) cropHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	query := r.URL.Query()
	srcURL, err := base64.StdEncoding.DecodeString(query.Get("src"))
	if err != nil {
		http.Error(w, "Invalid src parameter", http.StatusBadRequest)
		return
	}

	zoom, err := strconv.ParseFloat(query.Get("zoom"), 64)
	if err != nil {
		zoom = 1.0
	}

	size, err := strconv.ParseFloat(query.Get("size"), 64)
	if err != nil {
		size = 1.0
	}

	resp, err := http.Get(string(srcURL))
	if err != nil {
		http.Error(w, "Failed to download image", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	img, _, err := image.Decode(resp.Body)
	if err != nil {
		http.Error(w, "Failed to decode image", http.StatusInternalServerError)
		return
	}

	faces := fd.DetectFaces(img)
	rects := GetFaceRects(faces, 5.0)
	if len(rects) == 0 {
		decoded, err := base64.StdEncoding.DecodeString(defaultFaceBase64)
		if err != nil {
			http.Error(w, "Failed to decode default face image", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "image/jpeg")
		w.Header().Set("Content-Length", strconv.Itoa(len(decoded)))
		if _, err := w.Write(decoded); err != nil {
			log.Println("unable to write image.")
		}
		return
	}

	faceRect := rects[0]
	width, height := faceRect.Dx(), faceRect.Dy()

	newWidth := int(float64(width) * zoom)
	newHeight := int(float64(height) * zoom)

	centerX := faceRect.Min.X + width/2
	centerY := faceRect.Min.Y + height/2

	minX := centerX - newWidth/2
	minY := centerY - newHeight/2
	maxX := centerX + newWidth/2
	maxY := centerY + newHeight/2

	// Clamp to image bounds
	bounds := img.Bounds()
	minX = max(minX, bounds.Min.X)
	minY = max(minY, bounds.Min.Y)
	maxX = min(maxX, bounds.Max.X)
	maxY = min(maxY, bounds.Max.Y)

	croppedImg, err := cropImage(img, image.Rect(minX, minY, maxX, maxY))
	if err != nil {
		http.Error(w, "Failed to crop image", http.StatusInternalServerError)
		return
	}

	resizedImg := resize.Resize(uint(float64(newWidth)*size), uint(float64(newHeight)*size), croppedImg, resize.Lanczos3)

	buffer := new(bytes.Buffer)
	if err := jpeg.Encode(buffer, resizedImg, nil); err != nil {
		http.Error(w, "Failed to encode image", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "image/jpeg")
	w.Header().Set("Content-Length", strconv.Itoa(len(buffer.Bytes())))
	if _, err := w.Write(buffer.Bytes()); err != nil {
		log.Println("unable to write image.")
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	detector, err := NewFaceDetector(cascadeFile)
	if err != nil {
		log.Fatalf("Failed to create face detector: %v", err)
	}

	http.HandleFunc("/crop", detector.cropHandler)
	log.Println("Server started on :8081")
	log.Fatal(http.ListenAndServe(":8081", nil))
}
