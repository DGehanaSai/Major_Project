export default function UploadSoil() {
  return (
    <div className="card">
  <h2>Start Your Crop Analysis</h2>

  <input type="file" />
  <select>
    <option>Select soil type</option>
    <option>Loam</option>
    <option>Sandy Loam</option>
  </select>

  <select>
    <option>Select crop type</option>
    <option>Cotton</option>
    <option>Wheat</option>
  </select>

  <button>Start Analysis</button>
</div>

  );
}
