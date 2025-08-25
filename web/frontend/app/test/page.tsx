"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Upload } from "lucide-react";
import Image from "next/image";
import { Input } from "@/components/ui/input";
import {
  ReactCompareSlider,
  ReactCompareSliderImage,
} from "react-compare-slider";
import { div } from "motion/react-client";

export default function ImageProcessor() {
  const [file, setFile] = useState<File | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setPreview(URL.createObjectURL(uploadedFile));
      setProcessedImage(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true); // bật loading
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/process-image", {
        method: "POST",
        body: formData,
      });

      const blob = await res.blob();
      setProcessedImage(URL.createObjectURL(blob));
    } catch (error) {
      console.error("Lỗi xử lý ảnh:", error);
    } finally {
      setLoading(false); // tắt loading
    }
  };

  return (

    <div className="flex flex-col items-center gap-6 p-6">

      <div>
<h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-rose-600 via-pink-500 to-violet-600 bg-clip-text text-transparent">
  Xóa watermark với mô hình MNet
</h1>


        <p className="text-sm">Bạn có thể thử nghiệm một số hình ảnh sau <a className="text-blue-400" target="_blank" href="https://drive.google.com/drive/folders/1V8qKfjJ35BQ4XZeu83tt3gXoprfD3L8X?usp=sharing">tại đây</a></p>
      </div>

      {/* Ô chọn ảnh */}
      <Card className="w-full max-w-md border-dashed border-2">
        <CardContent className="flex flex-col items-center justify-center p-6">
          <Input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="cursor-pointer"
          />
          <Upload className="h-8 w-8 text-gray-500 mt-2" />
          <p className="text-gray-500 text-sm mt-2">Click để chọn</p>
        </CardContent>
      </Card>

      {preview && (
        <>
          <Button onClick={handleUpload} disabled={loading}>
            {loading ? (
              <>
                <svg
                  className="animate-spin h-4 w-4 mr-2 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8v4l3-3-3-3v4a12 12 0 00-12 12h4z"
                  ></path>
                </svg>
                Đang xử lý...
              </>
            ) : (
              "Upload và xử lý"
            )}
          </Button>

          <div className="flex flex-col md:flex-row gap-6 mt-4 items-start">
            {/* Ảnh gốc */}
            <div className="flex flex-col items-center">
              <h3 className="mb-2 font-medium">Ảnh gốc</h3>
              <Image
                src={preview}
                alt="Original"
                width={255}
                height={255}
                className="rounded-lg shadow"
              />
            </div>

            {/* Ảnh sau xử lý */}
            {processedImage && (
              <div className="flex flex-col items-center">
                <h3 className="mb-2 font-medium">Ảnh sau xử lý</h3>
                <Image
                  src={processedImage}
                  alt="Processed"
                  width={255}
                  height={255}
                  className="rounded-lg shadow"
                />
              </div>
            )}
          </div>

          {/* Compare slider 300x300 */}
          {processedImage && (
            <div className="mt-6 flex flex-col items-center">
              <h3 className="mb-2 font-medium text-center">So sánh</h3>
              <div className="w-[300px] h-[300px]">
                <ReactCompareSlider
                  itemOne={
                    <ReactCompareSliderImage src={preview} alt="Ảnh gốc" />
                  }
                  itemTwo={
                    <ReactCompareSliderImage
                      src={processedImage}
                      alt="Ảnh sau xử lý"
                    />
                  }
                  position={50}
                  style={{
                    width: "100%",
                    height: "100%",
                    borderRadius: "0.5rem",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
                  }}
                />
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
