package main

import (
	"bytes"
	_ "embed"
	"encoding/base64"
	"flag"
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
		log.Println("No faces detected, returning original image.")
		buffer := new(bytes.Buffer)
		if err := jpeg.Encode(buffer, img, nil); err != nil {
			http.Error(w, "Failed to encode original image", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "image/jpeg")
		w.Header().Set("Content-Length", strconv.Itoa(len(buffer.Bytes())))
		if _, err := w.Write(buffer.Bytes()); err != nil {
			log.Println("unable to write image.")
		}
		return
	}

	faceRect := rects[0]
	width, height := faceRect.Dx(), faceRect.Dy()

	// The crop area should be a square, expanded by the zoom factor.
	cropSide := int(float64(width) * zoom)

	centerX := faceRect.Min.X + width/2
	centerY := faceRect.Min.Y + height/2

	minX := centerX - cropSide/2
	minY := centerY - cropSide/2
	maxX := centerX + cropSide/2
	maxY := centerY + cropSide/2

	// Clamp to image bounds
	bounds := img.Bounds()
	minX = max(minX, bounds.Min.X)
	minY = max(minY, bounds.Min.Y)
	maxX = min(maxX, bounds.Max.X)
	maxY = min(maxY, bounds.Max.Y)

	// To avoid distortion, we must crop a square.
	// The side of the square is the smaller of the clamped width/height.
	cropWidth := maxX - minX
	cropHeight := maxY - minY
	squareSide := min(cropWidth, cropHeight)

	// Center the square crop within the clamped rectangle.
	finalMinX := minX + (cropWidth-squareSide)/2
	finalMinY := minY + (cropHeight-squareSide)/2
	finalMaxX := finalMinX + squareSide
	finalMaxY := finalMinY + squareSide

	croppedImg, err := cropImage(img, image.Rect(finalMinX, finalMinY, finalMaxX, finalMaxY))
	if err != nil {
		http.Error(w, "Failed to crop image", http.StatusInternalServerError)
		return
	}

	// Resize the square crop to the final size.
	// The final size is determined by the original crop-side and the size multiplier.
	finalSize := uint(float64(cropSide) * size)
	resizedImg := resize.Resize(finalSize, finalSize, croppedImg, resize.Lanczos3)

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
	port := flag.Int("port", 8037, "server port")
	flag.Parse()

	detector, err := NewFaceDetector(cascadeFile)
	if err != nil {
		log.Fatalf("Failed to create face detector: %v", err)
	}

	http.HandleFunc("/crop", detector.cropHandler)
	addr := fmt.Sprintf(":%d", *port)
	log.Printf("Server started on %s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}
