const API_URL = "http://127.0.0.1:8001/predict-cells";
const DEMO_IMAGE = "./public/demo-tray.png";

const demoBtn = document.querySelector("#demoBtn");
const cameraBtn = document.querySelector("#cameraBtn");
const captureBtn = document.querySelector("#captureBtn");
const cameraSelect = document.querySelector("#cameraSelect");
const fileInput = document.querySelector("#fileInput");
const heroFileInput = document.querySelector("#heroFileInput");
const imageStage = document.querySelector("#imageStage");
const captureCanvas = document.querySelector("#captureCanvas");
const billList = document.querySelector("#billList");
const cellList = document.querySelector("#cellList");
const totalValue = document.querySelector("#totalValue");
const statusText = document.querySelector("#statusText");
const statusDot = document.querySelector("#statusDot");
const priceGrid = document.querySelector("#priceGrid");
const heroDemoBtn = document.querySelector('[data-action="demo"]');
const subtotalValue = document.querySelector("#subtotalValue");
const taxValue = document.querySelector("#taxValue");
const checkoutTotalValue = document.querySelector("#checkoutTotalValue");
const checkoutBtn = document.querySelector("#checkoutBtn");
const voucherBtn = document.querySelector("#voucherBtn");
const paymentModal = document.querySelector("#paymentModal");
const closePaymentBtn = document.querySelector("#closePaymentBtn");
const paymentMethods = document.querySelector("#paymentMethods");
const paymentHint = document.querySelector("#paymentHint");
const receiptItems = document.querySelector("#receiptItems");
const receiptSubtotal = document.querySelector("#receiptSubtotal");
const receiptTax = document.querySelector("#receiptTax");
const receiptTotal = document.querySelector("#receiptTotal");
const confirmPaymentBtn = document.querySelector("#confirmPaymentBtn");

let latestCells = [];
let latestBill = { items: [], total: 0, currency: "VND" };
let selectedPaymentMethod = "cash";
let cameraStream = null;
let cameraVideo = null;
let cameraTimer = null;
let cameraBusy = false;
let cameraActive = false;
let cameraDevices = [];
let pendingStillCapture = false;

const CAMERA_INTERVAL_MS = 1800;
const TAX_RATE = 0.08;
const PAYMENT_METHODS = [
  {
    id: "cash",
    name: "Tiền mặt",
    detail: "Thanh toán tại quầy sau khi nhân viên xác nhận khay.",
    icon: "₫",
  },
  {
    id: "qr",
    name: "QR",
    detail: "Quét mã để chuyển khoản nhanh qua ứng dụng ngân hàng.",
    icon: "QR",
  },
  {
    id: "mastercard",
    name: "Mastercard",
    detail: "Thanh toán bằng thẻ tín dụng hoặc ghi nợ Mastercard.",
    icon: "MC",
  },
];
const MENU_ITEMS = [
  { name: "Cơm trắng", price: 10000, note: "Một giá tiền" },
  { name: "Đậu hũ sốt cà", price: 25000 },
  { name: "Cá hú kho", price: 30000 },
  { name: "Thịt kho trứng", price: 30000, note: "Thêm trứng +6.000 VND" },
  { name: "Thịt kho", price: 25000 },
  { name: "Canh chua có cá", price: 25000 },
  { name: "Canh chua không cá", price: 10000 },
  { name: "Sườn nướng", price: 30000 },
  { name: "Canh rau", price: 7000 },
  { name: "Rau xào", price: 10000 },
  { name: "Trứng chiên", price: 25000 },
];

function formatMoney(value, currency = "VND") {
  return `${Number(value || 0).toLocaleString("vi-VN")} ${currency}`;
}

function getBillTotals() {
  const total = Number(latestBill.total || 0);
  const subtotal = total ? Math.round(total / (1 + TAX_RATE)) : 0;
  return {
    subtotal,
    tax: total - subtotal,
    total,
    currency: latestBill.currency || "VND",
  };
}

function setStatus(kind, text) {
  statusDot.className = `status-dot ${kind}`;
  statusText.textContent = text;
}

function setBusy(isBusy) {
  demoBtn.disabled = isBusy;
  fileInput.disabled = isBusy;
  heroDemoBtn.disabled = isBusy;
  heroFileInput.disabled = isBusy;
  cameraBtn.disabled = isBusy && !cameraActive;
  captureBtn.disabled = false;
  captureBtn.hidden = false;
  cameraSelect.disabled = isBusy || !cameraDevices.length;
  cameraSelect.hidden = !cameraActive;
  checkoutBtn.disabled = isBusy || !latestBill.items.length || !latestBill.total;
}

function renderPriceGrid() {
  priceGrid.innerHTML = "";
  MENU_ITEMS.forEach((item) => {
    const card = document.createElement("article");
    card.className = "price-item";
    card.innerHTML = `
      <strong>${item.name}</strong>
      ${item.note ? `<small>${item.note}</small>` : ""}
      <span>${formatMoney(item.price)}</span>
    `;
    priceGrid.appendChild(card);
  });
}

function clearResults() {
  latestBill = { items: [], total: 0, currency: "VND" };
  billList.innerHTML = "";
  cellList.innerHTML = "";
  totalValue.textContent = "0 VND";
  renderCheckoutSummary();
}

function renderImage(file) {
  stopCamera();
  const url = URL.createObjectURL(file);
  imageStage.className = "image-stage";
  imageStage.innerHTML = "";

  const img = document.createElement("img");
  img.className = "preview-image";
  img.alt = "Ảnh khay ăn đang nhận diện";
  img.src = url;
  img.onload = () => {
    URL.revokeObjectURL(url);
    renderBoxes(img, latestCells);
  };
  imageStage.appendChild(img);
}

function renderVideo(stream) {
  imageStage.className = "image-stage";
  imageStage.innerHTML = "";

  cameraVideo = document.createElement("video");
  cameraVideo.className = "camera-video";
  cameraVideo.autoplay = true;
  cameraVideo.muted = true;
  cameraVideo.playsInline = true;
  cameraVideo.srcObject = stream;
  const enableCapture = () => {
    captureBtn.disabled = false;
    renderBoxes(cameraVideo, latestCells);
  };
  cameraVideo.addEventListener("loadedmetadata", () => {
    cameraVideo.play();
    enableCapture();
  });
  cameraVideo.addEventListener("canplay", enableCapture);
  imageStage.appendChild(cameraVideo);
}

function renderBoxes(img, cells) {
  imageStage.querySelectorAll(".box").forEach((box) => box.remove());
  const naturalWidth = img.naturalWidth || img.videoWidth;
  const naturalHeight = img.naturalHeight || img.videoHeight;
  if (!naturalWidth || !naturalHeight) return;

  const elementRect = img.getBoundingClientRect();
  const stageRect = imageStage.getBoundingClientRect();
  const elementRatio = elementRect.width / elementRect.height;
  const naturalRatio = naturalWidth / naturalHeight;

  let contentWidth = elementRect.width;
  let contentHeight = elementRect.height;
  let contentLeft = elementRect.left;
  let contentTop = elementRect.top;

  if (elementRatio > naturalRatio) {
    contentWidth = elementRect.height * naturalRatio;
    contentLeft = elementRect.left + (elementRect.width - contentWidth) / 2;
  } else {
    contentHeight = elementRect.width / naturalRatio;
    contentTop = elementRect.top + (elementRect.height - contentHeight) / 2;
  }

  const offsetLeft = contentLeft - stageRect.left;
  const offsetTop = contentTop - stageRect.top;
  const scaleX = contentWidth / naturalWidth;
  const scaleY = contentHeight / naturalHeight;

  cells.forEach((cell) => {
    const [x1, y1, x2, y2] = cell.box;
    const box = document.createElement("div");
    box.className = `box${cell.is_unknown ? " unknown" : ""}`;
    box.style.left = `${offsetLeft + x1 * scaleX}px`;
    box.style.top = `${offsetTop + y1 * scaleY}px`;
    box.style.width = `${(x2 - x1) * scaleX}px`;
    box.style.height = `${(y2 - y1) * scaleY}px`;

    const label = document.createElement("span");
    label.className = "box-label";
    label.textContent = cell.is_empty ? "empty" : cell.prediction.label;
    box.appendChild(label);
    imageStage.appendChild(box);
  });
}

function getPreviewElement() {
  return imageStage.querySelector("img") || imageStage.querySelector("video");
}

function renderPaymentMethods() {
  paymentMethods.innerHTML = "";
  PAYMENT_METHODS.forEach((method) => {
    const button = document.createElement("button");
    button.className = `payment-method${method.id === selectedPaymentMethod ? " selected" : ""}`;
    button.type = "button";
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", String(method.id === selectedPaymentMethod));
    button.dataset.method = method.id;
    button.innerHTML = `
      <span class="method-icon">${method.icon}</span>
      <span>
        <strong>${method.name}</strong>
        <small>${method.detail}</small>
      </span>
    `;
    paymentMethods.appendChild(button);
  });
  renderPaymentHint();
}

function renderPaymentHint() {
  const method = PAYMENT_METHODS.find((item) => item.id === selectedPaymentMethod);
  if (!method) return;

  if (method.id === "qr") {
    paymentHint.innerHTML = `
      <div class="qr-payment">
        <div class="qr-code" aria-hidden="true"></div>
        <div>
          <strong>Quét QR để thanh toán</strong>
          <span>Nội dung: TrayAI ${Date.now().toString().slice(-6)}</span>
        </div>
      </div>
    `;
    return;
  }

  if (method.id === "mastercard") {
    paymentHint.innerHTML = `
      <div class="card-payment">
        <div class="mastercard-mark" aria-hidden="true"><span></span><span></span></div>
        <div>
          <strong>Mastercard</strong>
          <span>Nhập thẻ trên máy POS hoặc cổng thanh toán của quầy.</span>
        </div>
      </div>
    `;
    return;
  }

  paymentHint.innerHTML = `
    <div class="cash-payment">
      <strong>Thanh toán tiền mặt</strong>
      <span>Nhân viên thu ngân xác nhận số tiền và hoàn tất hóa đơn.</span>
    </div>
  `;
}

function renderCheckoutSummary() {
  const totals = getBillTotals();
  subtotalValue.textContent = formatMoney(totals.subtotal, totals.currency);
  taxValue.textContent = formatMoney(totals.tax, totals.currency);
  checkoutTotalValue.textContent = formatMoney(totals.total, totals.currency);
  receiptSubtotal.textContent = formatMoney(totals.subtotal, totals.currency);
  receiptTax.textContent = formatMoney(totals.tax, totals.currency);
  receiptTotal.textContent = formatMoney(totals.total, totals.currency);

  receiptItems.innerHTML = "";
  if (!latestBill.items.length) {
    receiptItems.innerHTML = '<li class="muted">Chưa có món nào trong bill.</li>';
  } else {
    latestBill.items.forEach((item) => {
      const li = document.createElement("li");
      const name = document.createElement("span");
      const price = document.createElement("strong");
      name.textContent = item.name;
      price.textContent = formatMoney(item.price, totals.currency);
      li.append(name, price);
      receiptItems.appendChild(li);
    });
  }

  checkoutBtn.disabled = !latestBill.items.length || !latestBill.total;
}

function renderResult(result) {
  latestCells = result.cells || [];
  const currency = result.bill?.currency || "VND";
  latestBill = {
    items: result.bill?.items || [],
    total: Number(result.bill?.total || 0),
    currency,
  };
  totalValue.textContent = formatMoney(result.bill?.total, currency);
  renderCheckoutSummary();

  billList.innerHTML = "";
  const items = result.bill?.items || [];
  if (!items.length) {
    billList.innerHTML = '<li class="muted">Chưa có món nào được tính.</li>';
  } else {
    items.forEach((item) => {
      const li = document.createElement("li");
      li.className = "bill-item";
      li.innerHTML = `<span>${item.name}</span><strong>${formatMoney(item.price, currency)}</strong>`;
      billList.appendChild(li);
    });
  }

  cellList.innerHTML = "";
  latestCells.forEach((cell) => {
    const prediction = cell.prediction || {};
    const li = document.createElement("li");
    li.className = `cell-item${cell.is_unknown ? " unknown" : ""}`;
    const confidence = Math.round((prediction.confidence || 0) * 100);
    li.innerHTML = `
      <strong>${cell.is_empty ? "empty" : prediction.label}</strong>
      <span class="muted">Ô ${cell.cell_index} · ${confidence}%</span>
    `;
    cellList.appendChild(li);
  });

  const preview = getPreviewElement();
  if (preview) renderBoxes(preview, latestCells);
}

async function predictFile(file, { render = true } = {}) {
  if (render) {
    clearResults();
    latestCells = [];
    renderImage(file);
  }
  setBusy(true);
  setStatus("loading", "Đang nhận diện...");

  const formData = new FormData();
  formData.append("file", file, file.name || "tray.png");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP ${response.status}`);
    }
    const result = await response.json();
    renderResult(result);
    setStatus("ok", `Đã nhận diện ${latestCells.length} ô khay`);
  } catch (error) {
    setStatus("error", "Không gọi được API");
    cellList.innerHTML = `<li class="cell-item">${error.message}</li>`;
  } finally {
    setBusy(false);
  }
}

async function submitImage(file) {
  await predictFile(file, { render: true });
}

async function captureCameraFrame() {
  if (!cameraActive || cameraBusy || !cameraVideo?.videoWidth) return;

  cameraBusy = true;
  setStatus("loading", "Đang nhận diện từ camera...");
  try {
    captureCanvas.width = cameraVideo.videoWidth;
    captureCanvas.height = cameraVideo.videoHeight;
    const context = captureCanvas.getContext("2d");
    context.drawImage(cameraVideo, 0, 0, captureCanvas.width, captureCanvas.height);
    const blob = await new Promise((resolve) => {
      captureCanvas.toBlob(resolve, "image/jpeg", 0.9);
    });
    if (!blob) throw new Error("Không chụp được frame từ camera");

    const file = new File([blob], `camera-${Date.now()}.jpg`, { type: "image/jpeg" });
    const formData = new FormData();
    formData.append("file", file, file.name);

    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP ${response.status}`);
    }
    const result = await response.json();
    renderResult(result);
    setStatus("ok", `Camera đang nhận diện: ${latestCells.length} ô khay`);
  } catch (error) {
    setStatus("error", error.message);
  } finally {
    cameraBusy = false;
    if (pendingStillCapture) {
      pendingStillCapture = false;
      await captureCameraStill();
    }
  }
}

async function captureCameraStill() {
  if (cameraBusy) {
    pendingStillCapture = true;
    if (cameraTimer) {
      window.clearInterval(cameraTimer);
      cameraTimer = null;
    }
    setStatus("loading", "Sẽ chụp ảnh sau frame realtime hiện tại...");
    return;
  }

  if (!cameraActive) {
    const started = await startCamera(cameraSelect.value, { realtime: false });
    if (!started) return;
  }

  const videoReady = await waitForCameraFrame();
  if (!videoReady) {
    setStatus("error", "Camera chưa sẵn sàng, bấm Chụp ảnh lại sau 1 giây");
    return;
  }

  if (cameraTimer) {
    window.clearInterval(cameraTimer);
    cameraTimer = null;
  }

  cameraBusy = true;
  setBusy(true);
  setStatus("loading", "Đang chụp ảnh tĩnh...");

  try {
    captureCanvas.width = cameraVideo.videoWidth;
    captureCanvas.height = cameraVideo.videoHeight;
    const context = captureCanvas.getContext("2d");
    context.drawImage(cameraVideo, 0, 0, captureCanvas.width, captureCanvas.height);
    const blob = await new Promise((resolve) => {
      captureCanvas.toBlob(resolve, "image/jpeg", 0.95);
    });
    if (!blob) throw new Error("Không chụp được ảnh tĩnh từ camera");

    const file = new File([blob], `still-${Date.now()}.jpg`, { type: "image/jpeg" });
    setStatus("loading", "Đang nhận diện ảnh tĩnh...");
    await predictFile(file, { render: true });
  } catch (error) {
    cameraBusy = false;
    setBusy(false);
    setStatus("error", error.message);
  }
}

function waitForCameraFrame(timeoutMs = 3000) {
  const startedAt = performance.now();
  return new Promise((resolve) => {
    const check = () => {
      if (cameraVideo?.videoWidth && cameraVideo?.videoHeight) {
        resolve(true);
        return;
      }
      if (performance.now() - startedAt >= timeoutMs) {
        resolve(false);
        return;
      }
      window.setTimeout(check, 80);
    };
    check();
  });
}

function getCameraConstraints(deviceId = "") {
  const video = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
  };

  if (deviceId) {
    video.deviceId = { exact: deviceId };
  } else {
    video.facingMode = "environment";
  }

  return {
    video,
    audio: false,
  };
}

async function refreshCameraDevices(preferredDeviceId = cameraSelect.value) {
  if (!navigator.mediaDevices?.enumerateDevices) return;

  const devices = await navigator.mediaDevices.enumerateDevices();
  cameraDevices = devices.filter((device) => device.kind === "videoinput");
  cameraSelect.innerHTML = "";

  if (!cameraDevices.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Không tìm thấy camera";
    cameraSelect.appendChild(option);
    cameraSelect.disabled = true;
    return;
  }

  cameraDevices.forEach((camera, index) => {
    const option = document.createElement("option");
    option.value = camera.deviceId;
    option.textContent = camera.label || `Camera ${index + 1}`;
    cameraSelect.appendChild(option);
  });

  const selectedDeviceId = cameraDevices.some((camera) => camera.deviceId === preferredDeviceId)
    ? preferredDeviceId
    : cameraDevices[0].deviceId;
  cameraSelect.value = selectedDeviceId;
  cameraSelect.disabled = !cameraActive;
  cameraSelect.hidden = !cameraActive;
}

async function startCamera(deviceId = cameraSelect.value, { realtime = true } = {}) {
  stopCamera(false);
  clearResults();
  latestCells = [];
  setStatus("loading", "Đang mở camera...");

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Trình duyệt không hỗ trợ camera");
    }
    cameraStream = await navigator.mediaDevices.getUserMedia(getCameraConstraints(deviceId));
    const activeDeviceId = cameraStream.getVideoTracks()[0]?.getSettings().deviceId || deviceId;
    await refreshCameraDevices(activeDeviceId);
    cameraActive = true;
    cameraSelect.hidden = false;
    cameraSelect.disabled = !cameraDevices.length;
    captureBtn.hidden = false;
    captureBtn.disabled = false;
    cameraBtn.textContent = "Tắt camera";
    cameraBtn.classList.add("secondary");
    renderVideo(cameraStream);
    if (realtime) {
      setStatus("ok", "Camera realtime đang nhận diện...");
      cameraTimer = window.setInterval(captureCameraFrame, CAMERA_INTERVAL_MS);
      window.setTimeout(captureCameraFrame, 500);
    } else {
      setStatus("ok", "Camera đã mở. Đang chụp ảnh tĩnh...");
    }
    return true;
  } catch (error) {
    stopCamera(false);
    setStatus("error", `Không mở được camera: ${error.message}`);
    return false;
  }
}

function stopCamera(resetStage = true) {
  if (cameraTimer) {
    window.clearInterval(cameraTimer);
    cameraTimer = null;
  }
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  cameraVideo = null;
  cameraBusy = false;
  cameraActive = false;
  cameraBtn.textContent = "Mở camera";
  cameraBtn.classList.remove("secondary");
  captureBtn.hidden = false;
  captureBtn.disabled = false;
  cameraSelect.disabled = !cameraDevices.length;
  cameraSelect.hidden = true;
  if (resetStage) {
    imageStage.className = "image-stage empty";
    imageStage.innerHTML = "<p>Chọn ảnh hoặc dùng ảnh demo để bắt đầu.</p>";
    setStatus("idle", "Sẵn sàng");
  }
}

async function loadDemoImage() {
  const response = await fetch(DEMO_IMAGE);
  if (!response.ok) throw new Error("Không tìm thấy ảnh demo frontend/public/demo-tray.png");
  const blob = await response.blob();
  return new File([blob], "demo-tray.png", { type: blob.type || "image/png" });
}

demoBtn.addEventListener("click", async () => {
  try {
    const file = await loadDemoImage();
    await submitImage(file);
  } catch (error) {
    setStatus("error", error.message);
  }
});

heroDemoBtn.addEventListener("click", () => {
  demoBtn.click();
});

cameraBtn.addEventListener("click", () => {
  if (cameraActive) {
    stopCamera(true);
  } else {
    startCamera();
  }
});

captureBtn.addEventListener("click", async () => {
  await captureCameraStill();
});

cameraSelect.addEventListener("change", () => {
  if (cameraActive) {
    startCamera(cameraSelect.value);
  }
});

if (navigator.mediaDevices?.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", () => {
    refreshCameraDevices().catch(() => {});
  });
}

checkoutBtn.addEventListener("click", () => {
  if (!latestBill.items.length || !latestBill.total) {
    setStatus("error", "Chưa có bill để thanh toán");
    return;
  }
  renderCheckoutSummary();
  renderPaymentMethods();
  paymentModal.hidden = false;
  document.body.classList.add("modal-open");
});

closePaymentBtn.addEventListener("click", () => {
  paymentModal.hidden = true;
  document.body.classList.remove("modal-open");
});

paymentModal.addEventListener("click", (event) => {
  if (event.target === paymentModal) {
    paymentModal.hidden = true;
    document.body.classList.remove("modal-open");
  }
});

paymentMethods.addEventListener("click", (event) => {
  const button = event.target.closest("[data-method]");
  if (!button) return;
  selectedPaymentMethod = button.dataset.method;
  renderPaymentMethods();
});

confirmPaymentBtn.addEventListener("click", () => {
  const method = PAYMENT_METHODS.find((item) => item.id === selectedPaymentMethod);
  setStatus("ok", `Đã xác nhận thanh toán bằng ${method?.name || "phương thức đã chọn"}`);
  confirmPaymentBtn.textContent = "Đã xác nhận";
  window.setTimeout(() => {
    paymentModal.hidden = true;
    document.body.classList.remove("modal-open");
    confirmPaymentBtn.textContent = "Xác nhận thanh toán";
  }, 900);
});

voucherBtn.addEventListener("click", () => {
  setStatus("ok", "Mã khuyến mãi đã được ghi nhận cho bill hiện tại");
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !paymentModal.hidden) {
    paymentModal.hidden = true;
    document.body.classList.remove("modal-open");
  }
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) submitImage(file);
});

heroFileInput.addEventListener("change", () => {
  const file = heroFileInput.files?.[0];
  if (file) submitImage(file);
});

window.addEventListener("resize", () => {
  const preview = getPreviewElement();
  if (preview) renderBoxes(preview, latestCells);
});

renderPriceGrid();
renderPaymentMethods();
renderCheckoutSummary();
