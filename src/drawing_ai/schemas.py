"""Data contracts. Field aliases match the training-data spec exactly
(SPEC/Algorithm.pdf):

    {"Image ID": "IMG_001", "Tile ID": "Tile_001",
     "OBJECT/COMPONENT": [{"object_Bounding_Box": [x,y,w,h], "classification_type": "Ctype_1"}],
     "OCR": [{"content_bbox": [x,y,w,h], "text_extracted": "Ø25.4"}],
     "Annotation Arrow": [{"Points": [x1,y1,x2,y2], "class_id": "leader_line"}]}

Boxes are COCO ``[x, y, width, height]``. Tile annotations use tile-local pixel
coordinates; merged output uses canvas coordinates. ``confidence`` and
``source_tiles`` are optional extensions added by inference / merging.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Item(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


# Extension fields are declared after the spec fields in each class so the
# serialised JSON keeps the spec's key order.


class ObjectComponent(_Item):
    bbox: List[float] = Field(alias="object_Bounding_Box", min_length=4, max_length=4)
    classification_type: str
    confidence: Optional[float] = None
    source_tiles: Optional[List[str]] = None


class OCRItem(_Item):
    bbox: List[float] = Field(alias="content_bbox", min_length=4, max_length=4)
    text: str = Field(alias="text_extracted")
    confidence: Optional[float] = None
    source_tiles: Optional[List[str]] = None


class AnnotationArrow(_Item):
    points: List[float] = Field(alias="Points", min_length=4, max_length=4)
    class_id: str
    confidence: Optional[float] = None
    source_tiles: Optional[List[str]] = None


class TileAnnotation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_id: str = Field(alias="Image ID")
    tile_id: str = Field(alias="Tile ID")
    objects: List[ObjectComponent] = Field(default_factory=list, alias="OBJECT/COMPONENT")
    ocr: List[OCRItem] = Field(default_factory=list, alias="OCR")
    arrows: List[AnnotationArrow] = Field(default_factory=list, alias="Annotation Arrow")
    # Extension: tile placement on the canvas, [x, y, w, h].
    tile_box: Optional[List[int]] = None

    def dump(self) -> dict:
        return self.model_dump(by_alias=True, exclude_none=True)


class ImageAnnotation(BaseModel):
    """Image-level ground truth (coordinates in original image pixels).
    Feature engineering slices it into ``TileAnnotation`` objects."""

    model_config = ConfigDict(populate_by_name=True)

    image_id: str = Field(alias="Image ID")
    objects: List[ObjectComponent] = Field(default_factory=list, alias="OBJECT/COMPONENT")
    ocr: List[OCRItem] = Field(default_factory=list, alias="OCR")
    arrows: List[AnnotationArrow] = Field(default_factory=list, alias="Annotation Arrow")


class MergedAnnotation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    objects: List[ObjectComponent] = Field(default_factory=list, alias="OBJECT/COMPONENT")
    ocr: List[OCRItem] = Field(default_factory=list, alias="OCR")
    arrows: List[AnnotationArrow] = Field(default_factory=list, alias="Annotation Arrow")

    def dump(self) -> dict:
        return self.model_dump(by_alias=True, exclude_none=True)
