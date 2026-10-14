const DOT_RADIUS = 5;
const DEFAULT_OPACITY = 0.7;
const SELECTED_OPACITY = 1.0;
const COLOR_SCALE_SIZE = 10;
const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

function viewportToPx(val, axis) {
    const parsedValue = Number.parseFloat(val);
    if (!Number.isFinite(parsedValue)) {
        return 0;
    }
    if (String(val).endsWith("vw") || axis === "w") {
        return (parsedValue / 100) * document.documentElement.clientWidth;
    }
    if (String(val).endsWith("vh") || axis === "h") {
        return (parsedValue / 100) * document.documentElement.clientHeight;
    }
    return parsedValue;
}

function colorForClusterId(clusterId) {
    const numericClusterId = Number(clusterId);
    if (!Number.isFinite(numericClusterId)) {
        return colorScale(String(clusterId));
    }
    const normalizedClusterId = ((numericClusterId % COLOR_SCALE_SIZE) + COLOR_SCALE_SIZE) % COLOR_SCALE_SIZE;
    return colorScale(normalizedClusterId);
}

function syncInterpretationTabsColors() {
    document.querySelectorAll("#nav_tab [data-cluster-id]").forEach((button) => {
        const clusterId = button.getAttribute("data-cluster-id");
        button.style.color = colorForClusterId(clusterId);
    });
}

function paddedDomain(values) {
    const extent = d3.extent(values);
    if (!Number.isFinite(extent[0]) || !Number.isFinite(extent[1])) {
        return [-1, 1];
    }
    if (extent[0] === extent[1]) {
        return [extent[0] - 1, extent[1] + 1];
    }
    const padding = Math.max((extent[1] - extent[0]) * 0.08, 0.5);
    return [extent[0] - padding, extent[1] + padding];
}

function renderClusterGraph(config) {
    const container = d3.select(config.containerSelector);
    const containerNode = container.node();
    if (!containerNode) {
        return () => {};
    }

    const width = Math.max(
        viewportToPx(container.attr("width"), "w"),
        containerNode.clientWidth,
        480
    );
    const height = Math.max(
        viewportToPx(container.attr("height"), "h"),
        420
    );
    const margin = { top: 24, right: 28, bottom: 48, left: 64 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const clipId = `${config.containerSelector.replace("#", "")}-clip`;

    container.selectAll("*").remove();

    const svg = container
        .append("svg")
        .attr("class", "chart")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet")
        .style("display", "block")
        .style("max-width", "100%")
        .style("overflow", "hidden")
        .style("background", "#fff");

    svg.append("defs")
        .append("clipPath")
        .attr("id", clipId)
        .append("rect")
        .attr("width", innerWidth)
        .attr("height", innerHeight);

    svg.append("style").text(`
        .chart .axis path,
        .chart .axis line {
            stroke: #6c757d;
            shape-rendering: crispEdges;
        }

        .chart .axis text {
            fill: #495057;
            font-size: 12px;
        }

        .chart .axis-label {
            fill: #343a40;
            font-size: 13px;
            font-weight: 600;
        }

        .chart .plot-border {
            fill: none;
            stroke: #dee2e6;
            stroke-width: 1px;
        }

        .chart circle {
            opacity: ${DEFAULT_OPACITY};
            stroke: transparent;
            stroke-width: 0.6px;
        }

        .chart circle:hover {
            stroke: #111;
        }

        .chart .points-layer {
            will-change: transform;
        }
    `);

    const root = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    root.append("rect")
        .attr("class", "plot-border")
        .attr("width", innerWidth)
        .attr("height", innerHeight);

    const xAxisGroup = root.append("g")
        .attr("class", "axis axis-x")
        .attr("transform", `translate(0,${innerHeight})`);

    const yAxisGroup = root.append("g")
        .attr("class", "axis axis-y");

    root.append("text")
        .attr("class", "axis-label")
        .attr("x", innerWidth / 2)
        .attr("y", innerHeight + 38)
        .attr("text-anchor", "middle")
        .text("Axe X");

    root.append("text")
        .attr("class", "axis-label")
        .attr("transform", "rotate(-90)")
        .attr("x", -innerHeight / 2)
        .attr("y", -46)
        .attr("text-anchor", "middle")
        .text("Axe Y");

    const plotLayer = root.append("g")
        .attr("clip-path", `url(#${clipId})`);

    const zoomLayer = plotLayer.append("g")
        .attr("class", "points-layer");
    const labelLayer = plotLayer.append("g")
        .attr("class", "labels-layer");

    const tooltip = d3.select("body")
        .append("div")
        .style("opacity", 0)
        .style("z-index", "-200")
        .style("pointer-events", "none")
        .style("position", "absolute")
        .attr("class", `tooltip ${config.tooltipClass}`)
        .style("background-color", "white")
        .style("border", "solid")
        .style("border-width", "1px")
        .style("border-color", "#adb5bd")
        .style("border-radius", "5px")
        .style("padding", "5px");

    let labelsVisible = false;
    let currentTransform = d3.zoomIdentity;
    let pendingZoomEndFrame = null;
    let xScale = null;
    let yScale = null;

    function updateLabelsVisibility() {
        const opacity = labelsVisible ? "0.62" : "0";
        labelLayer.selectAll(`.${config.labelClass}`).attr("opacity", opacity);
    }

    function updateLabelPositions(transform) {
        if (!xScale || !yScale) {
            return;
        }
        const visiblePointRadius = DOT_RADIUS;
        const labelGap = 6;
        labelLayer.selectAll(`.${config.labelClass}`)
            .attr("x", (d) => transform.applyX(xScale(d.x)))
            .attr("y", (d) => transform.applyY(yScale(d.y)) - visiblePointRadius - labelGap)
            .attr("font-size", "13px");
    }

    function updatePointRadius(transform) {
        zoomLayer.selectAll("circle").attr("r", DOT_RADIUS / transform.k);
    }

    d3.dsv(";", config.csvUrl).then((data) => {
        const mappedData = data
            .map((d) => ({
                x: Number(d.Axe_X),
                y: Number(d.Axe_Y),
                pointColor: colorForClusterId(d.Cluster),
                name: d.Name,
                cluster: d.Cluster,
            }))
            .filter((d) => Number.isFinite(d.x) && Number.isFinite(d.y));

        if (mappedData.length === 0) {
            root.append("text")
                .attr("x", innerWidth / 2)
                .attr("y", innerHeight / 2)
                .attr("text-anchor", "middle")
                .attr("fill", "#6c757d")
                .text("Aucune donnée de visualisation disponible");
            return;
        }

        xScale = d3.scaleLinear()
            .domain(paddedDomain(mappedData.map((d) => d.x)))
            .nice()
            .range([0, innerWidth]);

        yScale = d3.scaleLinear()
            .domain(paddedDomain(mappedData.map((d) => d.y)))
            .nice()
            .range([innerHeight, 0]);

        function drawAxes(xAxisScale, yAxisScale) {
            xAxisGroup.call(d3.axisBottom(xAxisScale).ticks(Math.max(4, Math.floor(innerWidth / 110))));
            yAxisGroup.call(d3.axisLeft(yAxisScale).ticks(Math.max(4, Math.floor(innerHeight / 75))));
        }

        drawAxes(xScale, yScale);

        zoomLayer.selectAll("circle")
            .data(mappedData)
            .join("circle")
            .attr("cx", (d) => xScale(d.x))
            .attr("cy", (d) => yScale(d.y))
            .attr("r", DOT_RADIUS)
            .attr("fill", (d) => d.pointColor)
            .on("mouseover", function (event, d) {
                tooltip
                    .style("opacity", 1)
                    .style("z-index", "1000")
                    .text(d.name);
                d3.select(this)
                    .style("stroke", "black")
                    .style("opacity", SELECTED_OPACITY);
            })
            .on("mousemove", (event) => {
                tooltip
                    .style("left", `${event.pageX + 10}px`)
                    .style("top", `${event.pageY + 15}px`);
            })
            .on("mouseout", function () {
                tooltip
                    .style("opacity", 0)
                    .style("z-index", "-200");
                d3.select(this)
                    .style("stroke", "transparent")
                    .style("opacity", DEFAULT_OPACITY);
            });

        labelLayer.selectAll(`.${config.labelClass}`)
            .data(mappedData)
            .enter()
            .append("text")
            .attr("class", config.labelClass)
            .attr("pointer-events", "none")
            .attr("font-size", "13px")
            .attr("text-anchor", "middle")
            .attr("opacity", "0")
            .text((d) => d.name);
        updateLabelPositions(currentTransform);

        const zoom = d3.zoom()
            .extent([[0, 0], [innerWidth, innerHeight]])
            .translateExtent([[0, 0], [innerWidth, innerHeight]])
            .scaleExtent([1, 20])
            .on("zoom", ({ transform }) => {
                currentTransform = transform;
                tooltip.style("opacity", 0).style("z-index", "-200");
                zoomLayer.attr("transform", transform);
                updatePointRadius(transform);
                updateLabelPositions(transform);
            })
            .on("end", ({ transform }) => {
                currentTransform = transform;
                if (pendingZoomEndFrame) {
                    cancelAnimationFrame(pendingZoomEndFrame);
                }
                pendingZoomEndFrame = requestAnimationFrame(() => {
                    pendingZoomEndFrame = null;
                    drawAxes(transform.rescaleX(xScale), transform.rescaleY(yScale));
                    updatePointRadius(transform);
                    updateLabelPositions(transform);
                    updateLabelsVisibility();
                });
            });

        svg.call(zoom);
        updateLabelsVisibility();
    });

    return () => {
        const checkbox = document.getElementById(config.checkboxId);
        labelsVisible = Boolean(checkbox && checkbox.checked);
        updateLabelPositions(currentTransform);
        updateLabelsVisibility();
    };
}

document.querySelectorAll(".nav-link").forEach((button, i) => {
    button.style.color = colorScale(i);
});
syncInterpretationTabsColors();

window.toggleLabelsVisibility = renderClusterGraph({
    containerSelector: "#clusteringGraph",
    checkboxId: "checkBox",
    labelClass: "titrePerso",
    tooltipClass: "tooltip-standard",
    csvUrl: `${static_url}data/visualisation.csv?t=${Date.now()}`,
});
