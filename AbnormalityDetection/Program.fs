
// adapt from C# ml example: https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started/AnomalyDetection_PowerMeterReadings/PowerAnomalyDetection

open System
open Microsoft.ML
open Deedle
open FSharp.Plotly

// read data from csv file
let df = Frame.ReadCsv @".\data\power-export_min.csv"
         |> Frame.indexRowsDate "time"
let mlContext = MLContext(Nullable 0)

// Data Column Name
let dataCol = "ConsumptionDiffNormalized"
let convertSeriesToData (k:DateTime) (s:ObjectSeries<string>) = 
    {|
        name = s.GetAs<string> "name"
        time  = k
        ConsumptionDiffNormalized = s.GetAs<float32> dataCol
    |}

let buildTrainModel (mlc:MLContext) data = 
    // tuning parameters
    let PValueSize = 30
    let SeasonalitySize = 30
    let TrainingSize = 90
    let ConfidenceInterval = 98
    let trainingPipe = 
        mlc.Transforms.DetectSpikeBySsa(
            "Prediction",
            dataCol,
            confidence = ConfidenceInterval,
            pvalueHistoryLength = PValueSize,
            trainingWindowSize = TrainingSize,
            seasonalityWindowSize = SeasonalitySize
        )
    let trainedModel = trainingPipe.Fit(data)
    trainedModel

[<CLIMutable>]
type SpikePrediction = 
    { 
        Prediction : double[] 
    }

let detectAbnormality (mlc:MLContext) (model:ITransformer) data = 
    let transformedData = model.Transform(data)
    mlc.Data.CreateEnumerable<SpikePrediction>(transformedData,false)
    |> Seq.cast<SpikePrediction>        
    |> Seq.toArray

let dataView =
    df
    |> Frame.mapRows convertSeriesToData
    |> Series.values
    |> mlContext.Data.LoadFromEnumerable

let model = buildTrainModel mlContext dataView
let pred  = detectAbnormality mlContext model dataView
let t = df.RowKeys |> Seq.toArray
let abnKeys = 
    Array.zip t pred
    |> Array.filter (fun (_,p) -> p.Prediction.[0] = 1.0)
    |> Array.map fst
let abnPoints = 
    df.[dataCol] |> Series.getAll abnKeys
// do plotting
let lineChart = 
    Chart.Line( Series.observations df.[dataCol], Name = "Raw Data" )
let pointChart = 
    Chart.Point( Series.observations abnPoints, Color ="red", Name = "Abnormal" )

[<EntryPoint>]
let main argv =
    // combine charts and shows in browser
    Chart.Combine [lineChart;pointChart]
    |> Chart.Show
    0 // return an integer exit code