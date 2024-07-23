import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';

const RecordSessionSSL = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [chartData, setChartData] = useState(null);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const handleFileSubmit = async (event) => {
        event.preventDefault();
        if (selectedFile) {
            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const response = await fetch('http://localhost:5000/upload-ssl', {
                    method: 'POST',
                    body: formData,
                });

                if (response.ok) {
                    const data = await response.json();
                    const chartData = {
                        labels: data.time, // assuming the server returns a 'time' array
                        datasets: [
                            {
                                label: 'rPPG Signal',
                                data: data.rppg, // assuming the server returns an 'rppg' array
                                fill: false,
                                borderColor: 'rgba(75,192,192,1)',
                                tension: 0.1,
                            },
                        ],
                    };
                    setChartData(chartData);
                } else {
                    console.log('File upload failed');
                }
            } catch (error) {
                console.error('Error uploading file:', error);
            }
        } else {
            console.log('No file selected');
        }
    };

    return (
        <div className="container mt-5">
            <h1 className="text-center">Record Session Using SSL</h1>
            <div className="card mx-auto" style={{ maxWidth: '500px' }}>
                <div className="card-body">
                    <h2 className="card-title text-center">Upload MP4 File</h2>
                    <form onSubmit={handleFileSubmit}>
                        <div className="form-group">
                            <input
                                type="file"
                                accept="video/mp4"
                                className="form-control"
                                onChange={handleFileChange}
                            />
                        </div>
                        <button type="submit" className="btn btn-primary btn-block">Upload</button>
                    </form>
                </div>
            </div>
            {chartData && (
                <div className="mt-5">
                    <h2 className="text-center">rPPG Signal Graph</h2>
                    <Line data={chartData} />
                </div>
            )}
        </div>
    );
};

export default RecordSessionSSL;
