"""Database models for ahcore's manifest database."""

from enum import Enum as PyEnum
from typing import List

from sqlalchemy import Column, DateTime, Enum, Float, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, relationship


class CategoryEnum(str, PyEnum):
    FIT = "fit"
    VALIDATE = "validate"
    TEST = "test"
    PREDICT = "predict"


class Base(DeclarativeBase):
    __abstract__ = True
    # pylint: disable=E1102
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())


class Manifest(Base):
    """Manifest table."""

    __tablename__ = "manifest"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)

    patients: Mapped[List["Patient"]] = relationship("Patient", back_populates="manifest")
    split_definitions: Mapped[List["SplitDefinitions"]] = relationship("SplitDefinitions", back_populates="manifest")

    def __str__(self) -> str:
        return f"Manifest {self.name}"


class Patient(Base):
    """Patient table."""

    __tablename__ = "patient"
    id = Column(Integer, primary_key=True)
    manifest_id = Column(Integer, ForeignKey("manifest.id"))
    patient_code = Column(String, unique=True)

    manifest: Mapped["Manifest"] = relationship("Manifest", back_populates="patients")
    labels: Mapped[List["PatientLabels"]] = relationship("PatientLabels", back_populates="patient")
    images: Mapped[List["Image"]] = relationship("Image", back_populates="patient")
    splits: Mapped[List["Split"]] = relationship("Split", back_populates="patient")


class PatientLabels(Base):
    """Patient labels table."""

    __tablename__ = "patient_labels"
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patient.id"), nullable=False)
    key = Column(String, nullable=False)
    value = Column(String, nullable=False)

    patient: Mapped["Patient"] = relationship("Patient", back_populates="labels")

    __table_args__ = (UniqueConstraint("key", "patient_id", name="uq_patient_label_key"),)


class Image(Base):
    """Image table."""

    __tablename__ = "image"
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patient.id"), nullable=False)
    filename = Column(String, unique=True, nullable=False)
    reader = Column(String)
    width = Column(Integer)
    height = Column(Integer)
    mpp = Column(Float)

    patient: Mapped["Patient"] = relationship("Patient", back_populates="images")
    masks: Mapped[List["Mask"]] = relationship("Mask", back_populates="image")
    annotations: Mapped[List["ImageAnnotations"]] = relationship("ImageAnnotations", back_populates="image")
    labels: Mapped[List["ImageLabels"]] = relationship("ImageLabels", back_populates="image")
    image_caches: Mapped[List["ImageCache"]] = relationship("ImageCache", back_populates="image")


class Mask(Base):
    """Mask table."""

    __tablename__ = "mask"
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("image.id"), nullable=False)
    filename = Column(String, unique=True)
    reader = Column(String)

    image: Mapped["Image"] = relationship("Image", back_populates="masks")


class ImageAnnotations(Base):
    """Image annotations table."""

    __tablename__ = "image_annotations"
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("image.id"), nullable=False)
    filename = Column(String, unique=True)
    reader = Column(String)

    image: Mapped["Image"] = relationship("Image", back_populates="annotations")


class ImageLabels(Base):
    """Image labels table."""

    __tablename__ = "image_labels"
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("image.id"), nullable=False)
    key = Column(String, nullable=False)
    value = Column(String, nullable=False)

    image: Mapped["Image"] = relationship("Image", back_populates="labels")

    __table_args__ = (UniqueConstraint("key", "image_id", name="uq_image_label_key"),)


class ImageCache(Base):
    """Image cache table."""

    __tablename__ = "image_cache"
    id = Column(Integer, primary_key=True)
    cache_description_id = Column(Integer, ForeignKey("cache_description.id"))
    image_id = Column(Integer, ForeignKey("image.id"), nullable=False)
    filename = Column(String, unique=True, nullable=False)
    reader = Column(String)
    stitching_mode = Column(String)

    cache_description: Mapped["CacheDescription"] = relationship("CacheDescription", back_populates="image_caches")
    image: Mapped["Image"] = relationship("Image", back_populates="image_caches")


class CacheDescription(Base):
    """Cache description table."""

    __tablename__ = "cache_description"
    id = Column(Integer, primary_key=True)
    split_definition_id = Column(Integer, ForeignKey("split_definitions.id"))
    mpp = Column(Float)
    tile_size_width = Column(Integer)
    tile_size_height = Column(Integer)
    tile_overlap_width = Column(Integer)
    tile_overlap_height = Column(Integer)
    tile_mode = Column(String)
    crop = Column(Integer, default=False)  # using Integer for boolean for DB compatibility
    mask_threshold = Column(Float)
    grid_order = Column(String)

    image_caches: Mapped[List["ImageCache"]] = relationship("ImageCache", back_populates="cache_description")
    split_definitions: Mapped[List["SplitDefinitions"]] = relationship(
        "SplitDefinitions", back_populates="cache_description"
    )


class SplitDefinitions(Base):
    """Split definitions table."""

    __tablename__ = "split_definitions"
    id = Column(Integer, primary_key=True)
    manifest_id = Column(Integer, ForeignKey("manifest.id"))
    version = Column(String, nullable=False, unique=True)
    description = Column(String)

    manifest: Mapped["Manifest"] = relationship("Manifest", back_populates="split_definitions")
    splits: Mapped[List["Split"]] = relationship("Split", back_populates="split_definition")
    cache_description: Mapped["CacheDescription"] = relationship("CacheDescription", back_populates="split_definitions")


class Split(Base):
    """Split table."""

    __tablename__ = "split"
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patient.id"), nullable=False)
    split_definition_id = Column(Integer, ForeignKey("split_definitions.id"), nullable=False)
    category: Column[CategoryEnum] = Column(Enum(CategoryEnum), nullable=False)

    patient: Mapped["Patient"] = relationship("Patient", back_populates="splits")
    split_definition: Mapped["SplitDefinitions"] = relationship("SplitDefinitions", back_populates="splits")

    __table_args__ = (UniqueConstraint("split_definition_id", "patient_id", name="uq_patient_split_key"),)
