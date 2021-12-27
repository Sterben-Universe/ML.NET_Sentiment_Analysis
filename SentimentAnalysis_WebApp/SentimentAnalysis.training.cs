﻿﻿// This file was auto-generated by ML.NET Model Builder. 
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML;

namespace SentimentAnalysis_WebApp
{
    public partial class SentimentAnalysis
    {
        public static ITransformer RetrainPipeline(MLContext context, IDataView trainData)
        {
            var pipeline = BuildPipeline(context);
            var model = pipeline.Fit(trainData);

            return model;
        }

        /// <summary>
        /// build the pipeline that is used from model builder. Use this function to retrain model.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(new []{new InputOutputColumnPair(@"logged_in", @"logged_in"),new InputOutputColumnPair(@"ns", @"ns"),new InputOutputColumnPair(@"sample", @"sample"),new InputOutputColumnPair(@"split", @"split")})      
                                    .Append(mlContext.Transforms.ReplaceMissingValues(new []{new InputOutputColumnPair(@"rev_id", @"rev_id"),new InputOutputColumnPair(@"year", @"year")}))      
                                    .Append(mlContext.Transforms.Text.FeaturizeText(@"comment", @"comment"))      
                                    .Append(mlContext.Transforms.Concatenate(@"Features", new []{@"logged_in",@"ns",@"sample",@"split",@"rev_id",@"year",@"comment"}))      
                                    .Append(mlContext.Transforms.Conversion.MapValueToKey(@"Label", @"Label"))      
                                    .Append(mlContext.Transforms.NormalizeMinMax(@"Features", @"Features"))      
                                    .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(l1Regularization:1F,l2Regularization:1F,labelColumnName:@"Label",featureColumnName:@"Features"))      
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(@"PredictedLabel", @"PredictedLabel"));

            return pipeline;
        }
    }
}
