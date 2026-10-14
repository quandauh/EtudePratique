window.toggleLabelsVisibilityCompressed = renderClusterGraph({
    containerSelector: "#clusteringGraphCompressed",
    checkboxId: "checkBoxCompressed",
    labelClass: "titrePersoCompressed",
    tooltipClass: "tooltip-compressed",
    csvUrl: `${static_url}data/visualisation_compressed.csv?t=${Date.now()}`,
});
