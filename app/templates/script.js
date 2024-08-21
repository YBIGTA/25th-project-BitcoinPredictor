async function fetchData(startDate = null, endDate = null) {
    const response = await fetch('/predict');
    const json = await response.json();
    let data = json.data;

    // 날짜 범위가 설정되었을 때 데이터 필터링
    if (startDate && endDate) {
        data = data.filter(row => row.Date >= startDate && row.Date <= endDate);
    }

    return data;
}

async function updateChart(timeframe = 'day') {
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    const data = await fetchData(startDate, endDate);

    // 데이터가 충분한지 확인
    if (data.length === 0) {
        document.getElementById('candlestick-chart').innerHTML = "<p>No data available for the selected date range.</p>";
        return;
    }

    const dates = data.map(row => row.Date);
    const opens = data.map(row => row.Open);
    const highs = data.map(row => row.High);
    const lows = data.map(row => row.Low);
    const closes = data.map(row => row.Close);

    // 색상을 기준으로 데이터를 나누기 위한 기준 날짜 설정
    const splitDate = new Date("2024-07-10T00:00:00");

    const traceBefore = {
        x: dates.filter((date, i) => new Date(date) < splitDate),
        close: closes.filter((close, i) => new Date(dates[i]) < splitDate),
        decreasing: {line: {color: 'red'}},
        high: highs.filter((high, i) => new Date(dates[i]) < splitDate),
        increasing: {line: {color: 'green'}},
        line: {color: 'yellow'},  // 노란색으로 표시
        low: lows.filter((low, i) => new Date(dates[i]) < splitDate),
        open: opens.filter((open, i) => new Date(dates[i]) < splitDate),
        type: 'candlestick',
        xaxis: 'x',
        yaxis: 'y',
        name: 'Before July 10th'
    };

    const traceAfter = {
        x: dates.filter((date, i) => new Date(date) >= splitDate),
        close: closes.filter((close, i) => new Date(dates[i]) >= splitDate),
        decreasing: {line: {color: 'red'}},
        high: highs.filter((high, i) => new Date(dates[i]) >= splitDate),
        increasing: {line: {color: 'green'}},
        line: {color: 'blue'},  // 파란색으로 표시
        low: lows.filter((low, i) => new Date(dates[i]) >= splitDate),
        open: opens.filter((open, i) => new Date(dates[i]) >= splitDate),
        type: 'candlestick',
        xaxis: 'x',
        yaxis: 'y',
        name: 'After July 10th'
    };

    const layout = {
        title: 'Bitcoin Price',
        xaxis: {title: 'Date'},
        yaxis: {title: 'Price'},
        showlegend: true, // 범례 추가
        responsive: true  // 반응형 차트 설정
    };

    Plotly.newPlot('candlestick-chart', [traceBefore, traceAfter], layout);
}

// 페이지 로드 시 차트 초기화
updateChart();
