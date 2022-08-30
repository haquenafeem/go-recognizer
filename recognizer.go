package recognizer

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"os"

	goFace "github.com/Kagami/go-face"
)

// Data descriptor of the human face.
type Data struct {
	Id         string
	Descriptor goFace.Descriptor
}

// Face holds coordinates and descriptor of the human face.
type Face struct {
	Data
	Rectangle image.Rectangle
}

/*
A Recognizer creates face descriptors for provided images and
classifies them into categories.
*/
type Recognizer struct {
	Tolerance float32
	rec       *goFace.Recognizer
	UseCNN    bool
	UseGray   bool
	Dataset   []Data
}

/*
Init initialise a recognizer interface.
*/
func (_this *Recognizer) Init(path string) error {

	_this.Tolerance = 0.4
	_this.UseCNN = false
	_this.UseGray = true

	_this.Dataset = make([]Data, 0)

	rec, err := goFace.NewRecognizer(path)

	if err == nil {
		_this.rec = rec
	}

	return err

}

/*
Close frees resources taken by the Recognizer. Safe to call multiple
times. Don't use Recognizer after close call.
*/
func (_this *Recognizer) Close() {

	_this.rec.Close()

}

func (_this *Recognizer) addImageToDatasetAndReturnFaceData(path string, id string) (*Data, error) {
	file := path
	var err error

	if _this.UseGray {

		file, err = _this.createTempGrayFile(file, id)

		if err != nil {
			return nil, err
		}

		defer os.Remove(file)

	}

	var faces []goFace.Face

	if _this.UseCNN {
		faces, err = _this.rec.RecognizeFileCNN(file)
	} else {
		faces, err = _this.rec.RecognizeFile(file)
	}

	if err != nil {
		return nil, err
	}

	if len(faces) == 0 {
		return nil, errors.New("not a face on the image")
	}

	if len(faces) > 1 {
		return nil, errors.New("not a single face on the image")
	}

	f := Data{}
	f.Id = id
	f.Descriptor = faces[0].Descriptor

	_this.Dataset = append(_this.Dataset, f)

	return &f, nil
}

/*
AddImageToDataset add a sample image to the dataset
*/
func (_this *Recognizer) AddImageToDataset(path string, Id string) error {
	_, err := _this.addImageToDatasetAndReturnFaceData(path, Id)
	return err
}

/*
AddRawImageToDataset addd a sample golang image to the dataset
*/
func (_this *Recognizer) AddRawImageToDataset(img image.Image, id string) (*Data, error) {
	tmpFile := os.TempDir() + "/" + "123e4567-e89b-12d3-a456-426614174000.jpg"
	f, err := os.Create(tmpFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if err = jpeg.Encode(f, img, nil); err != nil {
		return nil, err
	}

	return _this.addImageToDatasetAndReturnFaceData(tmpFile, id)
}

/*
AddImageBytesToDataset addd a sample golang image to the dataset
*/
func (_this *Recognizer) AddImageBytesToDataset(imgBytes []byte, id string) (*Data, error) {
	img, _, err := image.Decode(bytes.NewReader(imgBytes))
	if err != nil {
		return nil, err
	}

	return _this.AddRawImageToDataset(img, id)
}

/*
AddSingleData adds a single data to the dataset
*/
func (_this *Recognizer) AddSingleData(d Data) {
	_this.Dataset = append(_this.Dataset, d)
}

/*
AddMultipleData adds a single data to the dataset
*/
func (_this *Recognizer) AddMultipleData(datas []Data) {
	_this.Dataset = append(_this.Dataset, datas...)
}

func (_this *Recognizer) RemoveFromDataset(id string) {
	index := -1
	for i, f := range _this.Dataset {
		if f.Id == id {
			index = i
			break
		}
	}

	if index == -1 {
		return
	}
	_this.Dataset = append(_this.Dataset[:index], _this.Dataset[index+1:]...)
	_this.SetSamples()
}

/*
SetSamples sets known descriptors so you can classify the new ones.
*/
func (_this *Recognizer) SetSamples() {

	var samples []goFace.Descriptor
	var avengers []int32

	for i, f := range _this.Dataset {
		samples = append(samples, f.Descriptor)
		avengers = append(avengers, int32(i))
	}

	_this.rec.SetSamples(samples, avengers)

}

/*
RecognizeSingle returns face if it's the only face on the image or nil otherwise.
Only JPEG format is currently supported.
*/
func (_this *Recognizer) RecognizeSingle(path string) (goFace.Face, error) {

	file := path
	var err error

	if _this.UseGray {

		file, err = _this.createTempGrayFile(file, "64ab59ac42d69274f06eadb11348969e")

		if err != nil {
			return goFace.Face{}, err
		}

		defer os.Remove(file)

	}

	var idFace *goFace.Face

	if _this.UseCNN {
		idFace, err = _this.rec.RecognizeSingleFileCNN(file)
	} else {
		idFace, err = _this.rec.RecognizeSingleFile(file)
	}

	if err != nil {
		return goFace.Face{}, fmt.Errorf("can't recognize: %v", err)

	}
	if idFace == nil {
		return goFace.Face{}, fmt.Errorf("not a single face on the image")
	}

	return *idFace, nil

}

/*
RecognizeMultiples returns all faces found on the provided image, sorted from
left to right. Empty list is returned if there are no faces, error is
returned if there was some error while decoding/processing image.
Only JPEG format is currently supported.
*/
func (_this *Recognizer) RecognizeMultiples(path string) ([]goFace.Face, error) {

	file := path
	var err error

	if _this.UseGray {

		file, err = _this.createTempGrayFile(file, "64ab59ac42d69274f06eadb11348969e")

		if err != nil {
			return nil, err
		}

		defer os.Remove(file)

	}

	var idFaces []goFace.Face

	if _this.UseCNN {
		idFaces, err = _this.rec.RecognizeFileCNN(file)
	} else {
		idFaces, err = _this.rec.RecognizeFile(file)
	}

	if err != nil {
		return nil, fmt.Errorf("can't recognize: %v", err)
	}

	return idFaces, nil

}

func (_this *Recognizer) RecognizeMultiplesFromImage(img image.Image) ([]goFace.Face, error) {
	uuid := "4209db13-5ac1-448c-8774-0c8ec51696a8"
	tmpFile := os.TempDir() + "/" + uuid + ".jpg"
	f, err := os.Create(tmpFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if err = jpeg.Encode(f, img, nil); err != nil {
		return nil, err
	}

	return _this.RecognizeMultiples(tmpFile)
}

/*
Classify returns all faces identified in the image. Empty list is returned if no match.
*/
func (_this *Recognizer) Classify(path string) ([]Face, error) {

	face, err := _this.RecognizeSingle(path)

	if err != nil {
		return nil, err
	}

	personID := _this.rec.ClassifyThreshold(face.Descriptor, _this.Tolerance)
	if personID < 0 {
		return nil, fmt.Errorf("can't classify")
	}

	facesRec := make([]Face, 0)
	aux := Face{Data: _this.Dataset[personID], Rectangle: face.Rectangle}
	facesRec = append(facesRec, aux)

	return facesRec, nil

}

func (_this *Recognizer) ClassifyWithImage(img image.Image) ([]Face, error) {
	tmpFile := os.TempDir() + "/" + "72c94a8e-a2fd-4fca-8869-ae957ba2e04a.jpg"
	f, err := os.Create(tmpFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if err = jpeg.Encode(f, img, nil); err != nil {
		return nil, err
	}

	return _this.Classify(tmpFile)
}

func (_this *Recognizer) ClassifyWithBytes(imgBytes []byte) ([]Face, error) {
	img, _, err := image.Decode(bytes.NewReader(imgBytes))
	if err != nil {
		return nil, err
	}

	return _this.ClassifyWithImage(img)
}

/*
ClassifyMultiples returns all faces identified in the image. Empty list is returned if no match.
*/
func (_this *Recognizer) ClassifyMultiples(path string) ([]Face, error) {

	faces, err := _this.RecognizeMultiples(path)

	if err != nil {
		return nil, fmt.Errorf("can't recognize: %v", err)
	}

	facesRec := make([]Face, 0)

	for _, f := range faces {

		personID := _this.rec.ClassifyThreshold(f.Descriptor, _this.Tolerance)
		if personID < 0 {
			continue
		}

		aux := Face{Data: _this.Dataset[personID], Rectangle: f.Rectangle}

		facesRec = append(facesRec, aux)

	}

	return facesRec, nil

}

func (_this *Recognizer) ClassifyMultiplesWithImage(img image.Image) ([]Face, error) {
	tmpFile := os.TempDir() + "/" + "72c94a8e-a2fd-4fca-8869-ae957ba2e04a.jpg"
	f, err := os.Create(tmpFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if err = jpeg.Encode(f, img, nil); err != nil {
		return nil, err
	}

	return _this.ClassifyMultiples(tmpFile)
}

func (_this *Recognizer) ClassifyMultiplesWithBytes(imgBytes []byte) ([]Face, error) {
	img, _, err := image.Decode(bytes.NewReader(imgBytes))
	if err != nil {
		return nil, err
	}

	return _this.ClassifyMultiplesWithImage(img)
}

/*
fileExists check se file exist
*/
func fileExists(FileName string) bool {
	file, err := os.Stat(FileName)
	return (err == nil) && !file.IsDir()
}

/*
jsonMarshal Marshal interface to array of byte
*/
func jsonMarshal(t interface{}) ([]byte, error) {
	buffer := &bytes.Buffer{}
	encoder := json.NewEncoder(buffer)
	encoder.SetEscapeHTML(false)
	err := encoder.Encode(t)
	return buffer.Bytes(), err
}
