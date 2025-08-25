import { MailIcon, MapPinIcon, PhoneIcon } from "lucide-react";
import Link from "next/link";

const Contact01Page = () => (
  <div className="min-h-screen flex items-center justify-center">
    <div className="text-center">
      <b className="text-muted-foreground">Liên hệ</b>
      <h2 className="mt-3 text-2xl md:text-4xl font-bold tracking-tight">
        Liên hệ với tôi
      </h2>
      <p className="mt-4 text-base sm:text-lg">
        Tôi luôn sẵn sàng trò chuyện và giải đáp các thắc mắc của bạn.
      </p>
      <div className="max-w-screen-xl mx-auto py-24 grid md:grid-cols-2 lg:grid-cols-3 gap-16 md:gap-10 px-6 md:px-0">
        <div className="text-center flex flex-col items-center">
          <div className="h-12 w-12 flex items-center justify-center bg-primary/10 text-primary rounded-full">
            <MailIcon />
          </div>
          <h3 className="mt-6 font-semibold text-xl">Email</h3>
          <p className="mt-2 text-muted-foreground">
            Bạn hãy ưu tiên gửi email tới tôi trước để có thể giải đáp kịp thời.
          </p>
          <Link
            className="mt-4 font-medium text-primary"
            href="mailto:lamhung13102005@gmail.com"
          >
            lamhung13102005@gmail.com
          </Link>
        </div>
        <div className="text-center flex flex-col items-center">
          <div className="h-12 w-12 flex items-center justify-center bg-primary/10 text-primary rounded-full">
            <MapPinIcon />
          </div>
          <h3 className="mt-6 font-semibold text-xl">Văn phòng</h3>
          <p className="mt-2 text-muted-foreground">
            Bạn có thể gặp tôi nếu cần
          </p>
          <Link
            className="mt-4 font-medium text-primary"
            href="https://map.google.com"
            target="_blank"
          >
            Láng Thượng, Đống Đa, Hà Nội
          </Link>
        </div>
        <div className="text-center flex flex-col items-center">
          <div className="h-12 w-12 flex items-center justify-center bg-primary/10 text-primary rounded-full">
            <PhoneIcon />
          </div>
          <h3 className="mt-6 font-semibold text-xl">Điện thoại</h3>
          <p className="mt-2 text-muted-foreground">Thứ Hai–Thứ Sáu từ 8:00 đến 17:00.</p>
          <Link
            className="mt-4 font-medium text-primary"
            href="/"
          >
            +84 379 181 JQK
          </Link>
        </div>
      </div>
    </div>
  </div>
);

export default Contact01Page;
